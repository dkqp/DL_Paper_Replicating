import torch
from torch import nn

class basic_conv_block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class inception_block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels_1by1: int,
        channels_3by3_reduction: int,
        channels_3by3: int,
        channels_5by5_reduction: int,
        channels_5by5: int,
        channels_proj: int
    ) -> None:
        super().__init__()
        self.branch1 = basic_conv_block(
            in_channels=in_channels,
            out_channels=channels_1by1,
            kernel_size=1
        )
        self.branch2 = nn.Sequential(
            basic_conv_block(
              in_channels=in_channels,
              out_channels=channels_3by3_reduction,
              kernel_size=1
            ),
            basic_conv_block(
              in_channels=channels_3by3_reduction,
              out_channels=channels_3by3,
              kernel_size=3,
              padding=1
            )
        )
        self.branch3 = nn.Sequential(
            basic_conv_block(
              in_channels=in_channels,
              out_channels=channels_5by5_reduction,
              kernel_size=1
            ),
            basic_conv_block(
              in_channels=channels_5by5_reduction,
              out_channels=channels_5by5,
              kernel_size=5,
              padding=2
            )
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(
              kernel_size=3,
              padding=1,
              stride=1
            ),
            basic_conv_block(
              in_channels=in_channels,
              out_channels=channels_proj,
              kernel_size=1
            )
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.concat([branch1, branch2, branch3, branch4], dim=1)

class auxiliary_block(nn.Module):
    def __init__(self, in_channels: int, num_features: int) -> None:
        super().__init__()
        self.average_pooling = nn.AvgPool2d(
            kernel_size=5,
            stride=3
        )
        self.reduction = basic_conv_block(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=1
        )
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=4*4*128,
                out_features=1024
            ),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(
            in_features=1024,
            out_features=num_features
        )

    def forward(self, x):
        x = self.average_pooling(x)
        x = self.reduction(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GoogLeNet(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.conv_pool_layer1 = nn.Sequential(
            basic_conv_block(
              in_channels=3,
              out_channels=64,
              kernel_size=7,
              padding=3,
              stride=2
            ),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                ceil_mode=True
            )
        )
        self.conv_pool_layer2 = nn.Sequential(
            basic_conv_block(
              in_channels=64,
              out_channels=64,
              kernel_size=1
            ),
            basic_conv_block(
              in_channels=64,
              out_channels=192,
              kernel_size=3,
              padding=1
            ),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                ceil_mode=True
            )
        )
        self.inception_layer3a = inception_block(
            in_channels=192,
            channels_1by1=64,
            channels_3by3_reduction=96,
            channels_3by3=128,
            channels_5by5_reduction=16,
            channels_5by5=32,
            channels_proj=32
        )
        self.inception_layer3b = inception_block(
            in_channels=256,
            channels_1by1=128,
            channels_3by3_reduction=128,
            channels_3by3=192,
            channels_5by5_reduction=32,
            channels_5by5=96,
            channels_proj=64
        )
        self.max_pool3 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            ceil_mode=True
        )
        self.inception_layer4a = inception_block(
            in_channels=480,
            channels_1by1=192,
            channels_3by3_reduction=96,
            channels_3by3=208,
            channels_5by5_reduction=16,
            channels_5by5=48,
            channels_proj=64
        )
        self.inception_layer4b = inception_block(
            in_channels=512,
            channels_1by1=160,
            channels_3by3_reduction=112,
            channels_3by3=224,
            channels_5by5_reduction=24,
            channels_5by5=64,
            channels_proj=64
        )
        self.inception_layer4c = inception_block(
            in_channels=512,
            channels_1by1=128,
            channels_3by3_reduction=128,
            channels_3by3=256,
            channels_5by5_reduction=24,
            channels_5by5=64,
            channels_proj=64
        )
        self.inception_layer4d = inception_block(
            in_channels=512,
            channels_1by1=112,
            channels_3by3_reduction=144,
            channels_3by3=288,
            channels_5by5_reduction=32,
            channels_5by5=64,
            channels_proj=64
        )
        self.inception_layer4e = inception_block(
            in_channels=528,
            channels_1by1=256,
            channels_3by3_reduction=160,
            channels_3by3=320,
            channels_5by5_reduction=32,
            channels_5by5=128,
            channels_proj=128
        )
        self.max_pool4 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            ceil_mode=True
        )
        self.inception_layer5a = inception_block(
            in_channels=832,
            channels_1by1=256,
            channels_3by3_reduction=160,
            channels_3by3=320,
            channels_5by5_reduction=32,
            channels_5by5=128,
            channels_proj=128
        )
        self.inception_layer5b = inception_block(
            in_channels=832,
            channels_1by1=384,
            channels_3by3_reduction=192,
            channels_3by3=384,
            channels_5by5_reduction=48,
            channels_5by5=128,
            channels_proj=128
        )
        self.average_pool = nn.AvgPool2d(
            kernel_size=7
        )
        self.dropout = nn.Dropout(p=0.4)
        self.classifier = nn.Linear(in_features=1024, out_features=num_features)
        self.auxiliary1 = auxiliary_block(in_channels=512, num_features=num_features)
        self.auxiliary2 = auxiliary_block(in_channels=528, num_features=num_features)

    def forward(self, x, aux: bool = True):
        x = self.conv_pool_layer1(x)

        x = self.conv_pool_layer2(x)

        x = self.inception_layer3a(x)
        x = self.inception_layer3b(x)
        x = self.max_pool3(x)

        x = self.inception_layer4a(x)
        aux1 = None
        if aux:
            aux1 = self.auxiliary1(x)
        x = self.inception_layer4b(x)
        x = self.inception_layer4c(x)
        x = self.inception_layer4d(x)
        aux2 = None
        if aux:
            aux2 = self.auxiliary2(x)
        x = self.inception_layer4e(x)
        x = self.max_pool3(x)

        x = self.inception_layer5a(x)
        x = self.inception_layer5b(x)
        x = self.average_pool(x)
        x = self.dropout(x)

        x = nn.Flatten()(x)
        x = self.classifier(x)

        return (aux1, aux2, x)
    