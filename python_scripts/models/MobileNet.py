import torch
from torch import nn

class DW_conv(nn.Module):
    def __init__(self, reduction, channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if reduction:
            self.DW = nn.Sequential(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=channels,
                    bias=False
                ),
                nn.BatchNorm2d(num_features=channels),
                nn.ReLU6()
            )
        else:
            self.DW = nn.Sequential(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=channels,
                    bias=False
                ),
                nn.BatchNorm2d(num_features=channels),
                nn.ReLU6()
            )

    def forward(self, X):
        return self.DW(X)

class PW_conv(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU6', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.PW = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels
            )
        )
        if activation == 'ReLU6':
            self.PW.add_module(
                '2',
                nn.ReLU6()
            )
        elif activation == 'linear':
            pass

    def forward(self, X):
        return self.PW(X)

class MobileNetV1(nn.Module):
    def __init__(self, num_classes, dropout=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sequence1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU6()
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
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=1024,
                out_features=num_classes
            )
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

class bottleneck_s1(nn.Module):
    def __init__(self, t, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual_connection = in_channels == out_channels
        self.bn = nn.Sequential(
            PW_conv(
                in_channels=in_channels,
                out_channels=in_channels * t,
                activation='ReLU6'
            ),
            DW_conv(
                reduction=False,
                channels=in_channels * t
            ),
            PW_conv(
                in_channels=in_channels * t,
                out_channels=out_channels,
                activation='linear'
            )
        )

    def forward(self, X):
        if self.residual_connection:
            return X + self.bn(X)
        return self.bn(X)

class bottleneck_s2(nn.Module):
    def __init__(self, t, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bn = nn.Sequential(
            PW_conv(
                in_channels=in_channels,
                out_channels=in_channels * t,
                activation='ReLU6'
            ),
            DW_conv(
                reduction=True,
                channels=in_channels * t
            ),
            PW_conv(
                in_channels=in_channels * t,
                out_channels=out_channels,
                activation='linear'
            )
        )

    def forward(self, X):
        return self.bn(X)

class bottlenecks(nn.Module):
    def __init__(self, t, in_channels, out_channels, n, reduction, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bottlenecks = []
        for i in range(n):
            if i == 0:
                if reduction:
                    self.bottlenecks.append(
                        bottleneck_s2(
                            t=t,
                            in_channels=in_channels,
                            out_channels=out_channels
                        )
                    )
                else:
                    self.bottlenecks.append(
                        bottleneck_s1(
                            t=t,
                            in_channels=in_channels,
                            out_channels=out_channels
                        )
                    )
            else:
                self.bottlenecks.append(
                    bottleneck_s1(
                        t=t,
                        in_channels=out_channels,
                        out_channels=out_channels
                    )
                )

        self.bottlenecks = nn.Sequential(
            *self.bottlenecks
        )

    def forward(self, X):
        return self.bottlenecks(X)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes, dropout=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sequence1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU6()
        )
        self.sequence2 = nn.Sequential(
            bottlenecks(
                t=1,
                in_channels=32,
                out_channels=16,
                n=1,
                reduction=False
            ),
            bottlenecks(6, 16, 24, 2, True),
            bottlenecks(6, 24, 32, 3, True),
            bottlenecks(6, 32, 64, 4, True),
            bottlenecks(6, 64, 96, 3, False),
            bottlenecks(6, 96, 160, 3, True),
            bottlenecks(6, 160, 320, 1, False),
        )
        self.sequence3 = nn.Sequential(
            nn.Conv2d(
                in_channels=320,
                out_channels=1280,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=1280),
            nn.ReLU6()
        )
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=1280,
                out_features=num_classes
            )
        )

    def forward(self, X):
        X = self.sequence1(X)
        X = self.sequence2(X)
        X = self.sequence3(X)
        X = self.avgpool(X)
        X = self.flatten(X)
        X = self.classifier(X)
        return X
