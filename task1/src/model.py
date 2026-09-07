import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU -> MaxPool."""

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


class CNN(nn.Module):
    """Modular CNN with progressive channel growth.

    Architecture: stacked ConvBlocks -> AdaptiveAvgPool -> FC classifier.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        dropout_rate: float = 0.5,
        channels: list[int] | None = None,
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
            nn.Linear(channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
