"""Deep CNN — deeper architecture with double-conv blocks.

Each stage uses two consecutive Conv-BN-ReLU layers before pooling,
giving the network more capacity to learn complex features at each
spatial resolution.

Config parameters (MODEL_PARAMS):
    channels (list[int]):
        Output channels per stage. Each stage has 2 conv layers.
        Default: [64, 128, 256, 512]

    dropout_rate (float):
        Dropout probability in the classifier head.
        Default: 0.5

    fc_hidden (int):
        Hidden layer size in the classifier.
        Default: 256

Example JSON config:
    {
        "MODEL_NAME": "deep_cnn",
        "MODEL_PARAMS": {
            "channels": [64, 128, 256, 512],
            "dropout_rate": 0.5,
            "fc_hidden": 256
        }
    }
"""

import torch.nn as nn


class DoubleConvBlock(nn.Module):
    """Two consecutive Conv-BN-ReLU layers followed by MaxPool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class DeepCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        channels: list[int] | None = None,
        dropout_rate: float = 0.5,
        fc_hidden: int = 256,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512]

        blocks = []
        prev_ch = in_channels
        for ch in channels:
            blocks.append(DoubleConvBlock(prev_ch, ch))
            prev_ch = ch
        self.features = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(channels[-1], fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
