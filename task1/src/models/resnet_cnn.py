"""ResNet CNN — Deep Residual Learning for Image Recognition.

Reference:
    He, K., Zhang, X., Ren, S., & Sun, J. (2016).
    "Deep Residual Learning for Image Recognition."
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
    https://arxiv.org/abs/1512.03385

Uses the standard ResNet architecture with Bottleneck blocks (conv1x1 → conv3x3 → conv1x1)
and skip connections. Supports ResNet-50, ResNet-101, and ResNet-152 via the `depth` parameter.

Config parameters (MODEL_PARAMS):
    depth (int):
        ResNet variant to use. Must be one of: 50, 101, 152.
        Controls the number of bottleneck blocks per stage:
            - 50:  [3, 4, 6,  3]   ~23.5M params
            - 101: [3, 4, 23, 3]   ~42.5M params
            - 152: [3, 8, 36, 3]   ~58.1M params
        Default: 50

    dropout_rate (float):
        Dropout probability before the final classification layer.
        Range: 0.0 – 1.0.
        Default: 0.5

    pretrained (bool):
        NOT supported in this custom implementation.
        For transfer learning, use torchvision.models directly.

Example JSON configs:
    ResNet-50:
    {
        "MODEL_NAME": "resnet_cnn",
        "MODEL_PARAMS": {
            "depth": 50,
            "dropout_rate": 0.5
        }
    }

    ResNet-101:
    {
        "MODEL_NAME": "resnet_cnn",
        "MODEL_PARAMS": {
            "depth": 101,
            "dropout_rate": 0.5
        }
    }

    ResNet-152:
    {
        "MODEL_NAME": "resnet_cnn",
        "MODEL_PARAMS": {
            "depth": 152,
            "dropout_rate": 0.3
        }
    }
"""

import torch.nn as nn

# Block configurations per ResNet depth: {depth: [blocks in stage1..4]}
RESNET_CONFIGS: dict[int, list[int]] = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}


class Bottleneck(nn.Module):
    """Bottleneck block (1x1 → 3x3 → 1x1) with skip connection.

    The output channels are `planes * expansion` (expansion=4).
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


class ResNetCNN(nn.Module):
    """ResNet-50 / ResNet-101 / ResNet-152 for image classification.

    Architecture follows the original paper:
        conv7x7 → BN → ReLU → MaxPool → 4 stages of Bottleneck blocks → AvgPool → FC
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        depth: int = 50,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        if depth not in RESNET_CONFIGS:
            raise ValueError(
                f"Unsupported ResNet depth={depth}. Choose from: {list(RESNET_CONFIGS.keys())}"
            )
        layers = RESNET_CONFIGS[depth]
        self.in_planes = 64

        # Stem: conv7x7 → BN → ReLU → MaxPool
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 4 residual stages
        self.stage1 = self._make_stage(64, layers[0], stride=1)
        self.stage2 = self._make_stage(128, layers[1], stride=2)
        self.stage3 = self._make_stage(256, layers[2], stride=2)
        self.stage4 = self._make_stage(512, layers[3], stride=2)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512 * Bottleneck.expansion, num_classes),
        )

        # Weight initialization (He et al.)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        out_channels = planes * Bottleneck.expansion

        if stride != 1 or self.in_planes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        blocks = [Bottleneck(self.in_planes, planes, stride, downsample)]
        self.in_planes = out_channels

        for _ in range(1, num_blocks):
            blocks.append(Bottleneck(self.in_planes, planes))

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.classifier(x)
        return x
