"""Hybrid CNN — image encoder + metadata branch with two classification heads.

Architecture (from eda/analysis/task3_analysis.md):

    image  → CNN backbone → flatten → v_img ─┐
                                               ├─ concat → v ─┬─ head → gender
    CSV    → embeddings → MLP       → v_meta ─┘               └─ head → occasion

    Loss = L_gender + L_occasion

Two CNN backbones are available:
    - "simple"  — stack of Conv-BN-ReLU-Pool blocks (lightweight baseline)
    - "resnet"  — ResNet with Bottleneck blocks (depth 50 / 101 / 152)

Metadata branch encodes three CSV columns as learned embeddings:
    - articleType   → Embedding(vocab, article_embed_dim)
    - masterCategory → Embedding(vocab, master_embed_dim)
    - baseColour    → Embedding(vocab, colour_embed_dim)

Metadata dropout: during training, v_meta is randomly zeroed with probability
``meta_dropout_p`` so the CNN learns to predict without metadata. At test time,
pass zero indices when metadata is unavailable.

Config parameters (MODEL_PARAMS):
    backbone (str):
        CNN backbone architecture for the image branch.
        Options: "simple", "resnet".
        Default: "simple"

    channels (list[int]):
        (simple backbone only) Channel progression per ConvBlock stage.
        Each stage: Conv2d → BatchNorm → ReLU → MaxPool.
        Default: [32, 64, 128, 256]

    depth (int):
        (resnet backbone only) ResNet variant. Must be one of: 50, 101, 152.
        Controls the number of bottleneck blocks per stage:
            - 50:  [3, 4, 6,  3]   ~23.5M params
            - 101: [3, 4, 23, 3]   ~42.5M params
            - 152: [3, 8, 36, 3]   ~58.1M params
        Default: 50

    article_embed_dim (int):
        Embedding dimension for articleType (highest Cramer's V with both targets).
        Default: 16

    master_embed_dim (int):
        Embedding dimension for masterCategory.
        Default: 8

    colour_embed_dim (int):
        Embedding dimension for baseColour.
        Default: 8

    meta_dim (int):
        Output dimension of the metadata MLP (v_meta size).
        Default: 64

    fc_hidden (int):
        Hidden dimension for each classification head.
        Default: 128

    dropout_rate (float):
        Dropout probability in classification heads.
        Range: 0.0 – 1.0.
        Default: 0.5

    meta_dropout_p (float):
        Probability of zeroing v_meta during training (for test-time fallback).
        Range: 0.0 – 1.0.
        Default: 0.2

Example JSON configs:
    Simple backbone:
    {
        "MODEL_NAME": "hybrid_cnn",
        "MODEL_PARAMS": {
            "backbone": "simple",
            "channels": [32, 64, 128, 256],
            "dropout_rate": 0.5,
            "meta_dropout_p": 0.2
        }
    }

    ResNet-50 backbone:
    {
        "MODEL_NAME": "hybrid_cnn",
        "MODEL_PARAMS": {
            "backbone": "resnet",
            "depth": 50,
            "dropout_rate": 0.5,
            "meta_dropout_p": 0.2
        }
    }

    ResNet-152 backbone:
    {
        "MODEL_NAME": "hybrid_cnn",
        "MODEL_PARAMS": {
            "backbone": "resnet",
            "depth": 152,
            "dropout_rate": 0.3,
            "meta_dropout_p": 0.2
        }
    }
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────
# Backbone building blocks
# ──────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Bottleneck(nn.Module):
    """Bottleneck block (1x1 → 3x3 → 1x1) with skip connection.

    Output channels = planes * expansion (expansion=4).
    """

    expansion = 4

    def __init__(self, in_channels: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        out_channels = planes * self.expansion

        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


# ResNet depth → [blocks in stage1..4]
RESNET_CONFIGS: dict[int, list[int]] = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}


# ──────────────────────────────────────────────────────────
# Backbone factory
# ──────────────────────────────────────────────────────────

def _build_simple_backbone(in_channels: int, channels: list[int]) -> tuple[nn.Module, int]:
    """Build simple CNN backbone. Returns (backbone, output_dim)."""
    blocks = []
    prev_ch = in_channels
    for ch in channels:
        blocks.append(ConvBlock(prev_ch, ch))
        prev_ch = ch
    backbone = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    return backbone, channels[-1]


def _build_resnet_backbone(in_channels: int, depth: int) -> tuple[nn.Module, int]:
    """Build ResNet backbone (no classifier). Returns (backbone, output_dim)."""
    if depth not in RESNET_CONFIGS:
        raise ValueError(f"Unsupported ResNet depth={depth}. Choose from: {list(RESNET_CONFIGS.keys())}")

    layers_cfg = RESNET_CONFIGS[depth]

    # We need a mutable counter for in_planes across _make_stage calls.
    # Using a list so the nested function can mutate it.
    in_planes = [64]

    def _make_stage(planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        out_channels = planes * Bottleneck.expansion

        if stride != 1 or in_planes[0] != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes[0], out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        blocks = [Bottleneck(in_planes[0], planes, stride, downsample)]
        in_planes[0] = out_channels

        for _ in range(1, num_blocks):
            blocks.append(Bottleneck(in_planes[0], planes))

        return nn.Sequential(*blocks)

    stem = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )

    stage1 = _make_stage(64, layers_cfg[0], stride=1)
    stage2 = _make_stage(128, layers_cfg[1], stride=2)
    stage3 = _make_stage(256, layers_cfg[2], stride=2)
    stage4 = _make_stage(512, layers_cfg[3], stride=2)

    backbone = nn.Sequential(stem, stage1, stage2, stage3, stage4, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    out_dim = 512 * Bottleneck.expansion  # 2048
    return backbone, out_dim


# ──────────────────────────────────────────────────────────
# Hybrid model
# ──────────────────────────────────────────────────────────

class HybridCNN(nn.Module):
    """Two-head hybrid model: CNN (image) + ANN (metadata) → gender + occasion.

    Args:
        in_channels: Image input channels (3 for RGB).
        num_gender_classes: Number of gender classes (default 5).
        num_occasion_classes: Number of occasion/usage classes (default 4).
        num_article_types: Vocabulary size for articleType embedding.
        num_master_categories: Vocabulary size for masterCategory embedding.
        num_colours: Vocabulary size for baseColour embedding.
        backbone: CNN backbone type — "simple" or "resnet".
        channels: (simple only) Channel progression per ConvBlock stage.
        depth: (resnet only) ResNet depth — 50, 101, or 152.
        article_embed_dim: Embedding dimension for articleType.
        master_embed_dim: Embedding dimension for masterCategory.
        colour_embed_dim: Embedding dimension for baseColour.
        meta_dim: Output dimension of the metadata MLP.
        fc_hidden: Hidden dimension for each classification head.
        dropout_rate: Dropout rate in classification heads.
        meta_dropout_p: Probability of zeroing v_meta during training.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_gender_classes: int = 5,
        num_occasion_classes: int = 4,
        num_article_types: int = 126,
        num_master_categories: int = 8,
        num_colours: int = 47,
        backbone: str = "simple",
        channels: list[int] | None = None,
        depth: int = 50,
        article_embed_dim: int = 16,
        master_embed_dim: int = 8,
        colour_embed_dim: int = 8,
        meta_dim: int = 64,
        fc_hidden: int = 128,
        dropout_rate: float = 0.5,
        meta_dropout_p: float = 0.2,
    ):
        super().__init__()

        # ── CNN backbone (image encoder) ──
        if backbone == "simple":
            if channels is None:
                channels = [32, 64, 128, 256]
            self.cnn, cnn_out_dim = _build_simple_backbone(in_channels, channels)
        elif backbone == "resnet":
            self.cnn, cnn_out_dim = _build_resnet_backbone(in_channels, depth)
        else:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose from: simple, resnet")

        # ── Metadata branch (embedding + MLP) ──
        self.embed_article = nn.Embedding(num_article_types, article_embed_dim, padding_idx=0)
        self.embed_master = nn.Embedding(num_master_categories, master_embed_dim, padding_idx=0)
        self.embed_colour = nn.Embedding(num_colours, colour_embed_dim, padding_idx=0)

        embed_total = article_embed_dim + master_embed_dim + colour_embed_dim
        self.meta_mlp = nn.Sequential(
            nn.Linear(embed_total, meta_dim),
            nn.ReLU(inplace=True),
        )
        self.meta_dropout_p = meta_dropout_p

        # ── Fusion + classification heads ──
        fusion_dim = cnn_out_dim + meta_dim

        self.gender_head = nn.Sequential(
            nn.Linear(fusion_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden, num_gender_classes),
        )

        self.occasion_head = nn.Sequential(
            nn.Linear(fusion_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden, num_occasion_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        image: torch.Tensor,
        meta_article: torch.Tensor,
        meta_master: torch.Tensor,
        meta_colour: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            image: (B, C, H, W) image tensor.
            meta_article: (B,) long tensor — articleType vocab index.
            meta_master: (B,) long tensor — masterCategory vocab index.
            meta_colour: (B,) long tensor — baseColour vocab index.

        Returns:
            (gender_logits, occasion_logits) — each (B, num_classes).
        """
        # Image branch
        v_img = self.cnn(image)  # (B, cnn_out_dim)

        # Metadata branch
        e_article = self.embed_article(meta_article)
        e_master = self.embed_master(meta_master)
        e_colour = self.embed_colour(meta_colour)
        v_meta = self.meta_mlp(torch.cat([e_article, e_master, e_colour], dim=1))  # (B, meta_dim)

        # Metadata dropout — zero out v_meta during training so CNN
        # learns to predict without metadata (test-time fallback).
        if self.training and self.meta_dropout_p > 0:
            mask = torch.bernoulli(
                torch.full((v_meta.size(0), 1), 1 - self.meta_dropout_p, device=v_meta.device)
            )
            v_meta = v_meta * mask

        # Fusion
        v = torch.cat([v_img, v_meta], dim=1)  # (B, cnn_out_dim + meta_dim)

        gender_logits = self.gender_head(v)
        occasion_logits = self.occasion_head(v)

        return gender_logits, occasion_logits
