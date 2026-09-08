"""Light CNN — minimal model for fast prototyping and debugging.

Uses depthwise-separable convolutions to reduce parameter count
while maintaining reasonable accuracy. Good for quick iteration
or resource-constrained environments.

Config parameters (MODEL_PARAMS):
    channels (list[int]):
        Output channels per stage.
        Default: [16, 32, 64]
        Keep small for speed; increase for accuracy.

    dropout_rate (float):
        Dropout probability in the classifier head.
        Default: 0.3

    fc_hidden (int):
        Hidden layer size in the classifier.
        Default: 64

Example JSON config:
    {
        "MODEL_NAME": "light_cnn",
        "MODEL_PARAMS": {
            "channels": [16, 32, 64],
            "dropout_rate": 0.3,
            "fc_hidden": 64
        }
    }
"""

import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class LightCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        channels: list[int] | None = None,
        dropout_rate: float = 0.3,
        fc_hidden: int = 64,
    ):
        super().__init__()
        if channels is None:
            channels = [16, 32, 64]

        blocks = []
        prev_ch = in_channels
        for ch in channels:
            blocks.append(DepthwiseSeparableConv(prev_ch, ch))
            prev_ch = ch
        self.features = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(channels[-1], fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
