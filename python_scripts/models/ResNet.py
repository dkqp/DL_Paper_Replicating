import torch
from torch import nn

class conv_residual_bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        inner_channel_1: int,
        inner_channel_2: int,
        inner_channel_3: int,
        reduce: bool,
        first_iter: bool
    ) -> None:
        super().__init__()
        if reduce:
            stride = 2
        else:
            stride = 1
        self.conv_sequence = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=inner_channel_1,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=inner_channel_1),
            nn.Conv2d(
                in_channels=inner_channel_1,
                out_channels=inner_channel_2,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm2d(num_features=inner_channel_2),
            nn.Conv2d(
                in_channels=inner_channel_2,
                out_channels=inner_channel_3,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=inner_channel_3),
            nn.ReLU(inplace=True)
        )

        self.dim_match_conv = None
        if first_iter:
            self.dim_match_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=inner_channel_3,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(num_features=inner_channel_3),
            )

    def forward(self, x):
        skip = x
        x = self.conv_sequence(x)
        if self.dim_match_conv:
            skip = self.dim_match_conv(skip)
        return x + skip

class Resnet152(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                padding=3,
                stride=2,
                bias=False
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                padding=1,
                stride=2
            )
        )
        self.conv2 = self.conv_iter(
            conv_name='conv2',
            in_channels=64,
            inner_channel_1=64,
            inner_channel_2=64,
            inner_channel_3=256,
            iter_num=3,
            reduce=False
        )
        self.conv3 = self.conv_iter(
            conv_name='conv3',
            in_channels=256,
            inner_channel_1=128,
            inner_channel_2=128,
            inner_channel_3=512,
            iter_num=8
        )
        self.conv4 = self.conv_iter(
            conv_name='conv4',
            in_channels=512,
            inner_channel_1=256,
            inner_channel_2=256,
            inner_channel_3=1024,
            iter_num=36
        )
        self.conv5 = self.conv_iter(
            conv_name='conv5',
            in_channels=1024,
            inner_channel_1=512,
            inner_channel_2=512,
            inner_channel_3=2048,
            iter_num=3
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.classifier(x)
        return x

    def conv_iter(
        self,
        conv_name: str,
        in_channels: int,
        inner_channel_1: int,
        inner_channel_2: int,
        inner_channel_3: int,
        iter_num: int,
        reduce: bool = True
    ):
        convN = nn.Sequential()
        prev_channels = in_channels
        for i in range(iter_num):
            if i == 0 and reduce:
                reduce = True
            else:
                reduce = False

            if i == 0:
                first_iter = True
            else:
                first_iter = False

            convN.add_module(conv_name + '_' + str(i + 1), conv_residual_bottleneck(
                in_channels=prev_channels,
                inner_channel_1=inner_channel_1,
                inner_channel_2=inner_channel_2,
                inner_channel_3=inner_channel_3,
                reduce=reduce,
                first_iter=first_iter
            ))
            prev_channels = inner_channel_3
        return convN