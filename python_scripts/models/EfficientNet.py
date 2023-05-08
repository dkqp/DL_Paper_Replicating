import torch
from torch import nn

class DW_conv(nn.Module):
    def __init__(self, reduction, channels, kernel_size=3, activation='ReLU6', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if activation == 'ReLU6':
            self.activation = nn.ReLU6
        elif activation == 'SiLU':
            self.activation = nn.SiLU
        else:
            self.activation = nn.ReLU

        if reduction:
            self.stride = 2
        else:
            self.stride = 1

        self.DW = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=kernel_size//2,
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(num_features=channels),
            self.activation()
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
        elif activation == 'SiLU':
            self.PW.add_module(
                '2',
                nn.SiLU()
            )
        elif activation == 'linear':
            pass

    def forward(self, X):
        return self.PW(X)

class SE_conv(nn.Module):
    def __init__(self, in_channels, r=0.25, activation='ReLU6', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.se_channels = int(in_channels * r)
        if activation == 'ReLU6':
            self.activation = nn.ReLU6
        elif activation == 'SiLU':
            self.activation = nn.SiLU
        else:
            self.activation = nn.ReLU

        self.SE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.se_channels,
                kernel_size=1,
            ),
            self.activation(),
            nn.Conv2d(
                in_channels=self.se_channels,
                out_channels=in_channels,
                kernel_size=1,
            ),
            nn.Sigmoid()
        )

    def forward(self, X):
        return X * self.SE(X)

class bottleneck(nn.Module):
    def __init__(self, t, in_channels, out_channels, reduction, kernel_size, activation, se=0.25, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual_connection = (in_channels == out_channels) and not reduction

        self.bn = nn.Sequential()
        num = 0
        if t > 1:
            self.bn.add_module(
                str(num),
                PW_conv(
                    in_channels=in_channels,
                    out_channels=in_channels * t,
                    activation=activation
                )
            )
            num += 1

        self.bn.add_module(
            str(num),
            DW_conv(
                reduction=reduction,
                channels=in_channels * t,
                kernel_size=kernel_size,
                activation=activation
            )
        )
        num += 1

        if se > 0:
            self.bn.add_module(
                str(num),
                SE_conv(
                    in_channels=in_channels * t,
                    r=se,
                    activation=activation
                )
            )
            num += 1

        self.bn.add_module(
            str(num),
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

class bottlenecks(nn.Module):
    def __init__(self, t, in_channels, out_channels, n, reduction, kernel_size, activation, se, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bottlenecks = []
        self.reduction = reduction
        for i in range(n):
            if i != 0:
                self.reduction = False
                in_channels = out_channels
            self.bottlenecks.append(
                bottleneck(
                    t=t,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    reduction=self.reduction,
                    kernel_size=kernel_size,
                    activation=activation,
                    se=se
                )
            )

        self.bottlenecks = nn.Sequential(
            *self.bottlenecks
        )

    def forward(self, X):
        return self.bottlenecks(X)

class EfficientNetB0(nn.Module):
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
            nn.SiLU()
        )
        self.sequence2 = nn.Sequential(
            bottlenecks(
                t=1,
                in_channels=32,
                out_channels=16,
                n=1,
                reduction=False,
                kernel_size=3,
                activation='SiLU',
                se=0.25
            ),
            bottlenecks(6, 16, 24, 2, True, 3, 'SiLU', 0.0417),
            bottlenecks(6, 24, 40, 2, True, 5, 'SiLU', 0.0417),
            bottlenecks(6, 40, 80, 3, True, 3, 'SiLU', 0.0417),
            bottlenecks(6, 80, 112, 3, False, 5, 'SiLU', 0.0417),
            bottlenecks(6, 112, 192, 4, True, 5, 'SiLU', 0.0417),
            bottlenecks(6, 192, 320, 1, False, 3, 'SiLU', 0.0417),
        )
        self.sequence3 = nn.Sequential(
            nn.Conv2d(
                in_channels=320,
                out_channels=1280,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=1280),
            nn.SiLU()
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
