"""Simple CNN — lightweight baseline model.

A straightforward stack of Conv-BN-ReLU-Pool blocks followed by
adaptive average pooling and a fully-connected classifier.

Config parameters (MODEL_PARAMS):
    channels (list[int]):
        Number of output channels for each convolutional block.
        More blocks = deeper network; larger values = wider layers.
        Default: [32, 64, 128, 256]
        Example: [16, 32, 64] for a shallower/faster variant.

    dropout_rate (float):
        Dropout probability applied in the classifier head.
        Range: 0.0 – 1.0. Higher = more regularization.
        Default: 0.5

    fc_hidden (int):
        Number of neurons in the hidden FC layer before the output.
        Default: 128

Example JSON config:
    {
        "MODEL_NAME": "simple_cnn",
        "MODEL_PARAMS": {
            "channels": [32, 64, 128, 256],
            "dropout_rate": 0.5,
            "fc_hidden": 128
        }
    }
"""

import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SimpleCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        channels: list[int] | None = None,
        dropout_rate: float = 0.5,
        fc_hidden: int = 128,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        blocks = []
        prev_ch = in_channels
        for ch in channels:
            blocks.append(ConvBlock(prev_ch, ch, pool=True))
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
