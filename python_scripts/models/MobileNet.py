import torch
from torch import nn

class DW_conv(nn.Module):
    def __init__(self, reduction, channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reduction = reduction
        self.DW_s1 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels
            ),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU()
        )
        self.DW_s2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=channels
            ),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU()
        )

    def forward(self, X):
        if self.reduction:
            return self.DW_s2(X)
        return self.DW_s1(X)

class PW_conv(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.PW = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(
                num_features=out_channels
            ),
            nn.ReLU()
        )

    def forward(self, X):
        return self.PW(X)

class MobileNetV1(nn.Module):
    def __init__(self, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sequence1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )
        self.sequence2 = nn.Sequential(
            DW_conv(reduction=False, channels=32),
            PW_conv(in_channels=32, out_channels=64),
            DW_conv(reduction=True, channels=64),
            PW_conv(in_channels=64, out_channels=128)
        )
        self.sequence3 = nn.Sequential(
            DW_conv(reduction=False, channels=128),
            PW_conv(in_channels=128, out_channels=128),
            DW_conv(reduction=True, channels=128),
            PW_conv(in_channels=128, out_channels=256)
        )
        self.sequence4 = nn.Sequential(
            DW_conv(reduction=False, channels=256),
            PW_conv(in_channels=256, out_channels=256),
            DW_conv(reduction=True, channels=256),
            PW_conv(in_channels=256, out_channels=512)
        )
        self.sequence5 = nn.Sequential(
            DW_conv(reduction=False, channels=512),
            PW_conv(in_channels=512, out_channels=512),
            DW_conv(reduction=False, channels=512),
            PW_conv(in_channels=512, out_channels=512),
            DW_conv(reduction=False, channels=512),
            PW_conv(in_channels=512, out_channels=512),
            DW_conv(reduction=False, channels=512),
            PW_conv(in_channels=512, out_channels=512),
            DW_conv(reduction=False, channels=512),
            PW_conv(in_channels=512, out_channels=512),
        )
        self.sequence6 = nn.Sequential(
            DW_conv(reduction=True, channels=512),
            PW_conv(in_channels=512, out_channels=1024),
            DW_conv(reduction=False, channels=1024),
            PW_conv(in_channels=1024, out_channels=1024),
        )
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(
            in_features=1024,
            out_features=num_classes
        )

    def forward(self, X):
        X = self.sequence1(X)
        X = self.sequence2(X)
        X = self.sequence3(X)
        X = self.sequence4(X)
        X = self.sequence5(X)
        X = self.sequence6(X)
        X = self.avgpool(X)
        X = self.flatten(X)
        X = self.classifier(X)
        return X