import torch
from torch import nn

class VGG16(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            self.make_conv2d(in_channels=3, out_channels=64),
            torch.nn.ReLU(inplace=True),
            self.make_conv2d(in_channels=64, out_channels=64),
            torch.nn.ReLU(inplace=True),
            self.make_maxpool2d(),
            self.make_conv2d(in_channels=64, out_channels=128),
            torch.nn.ReLU(inplace=True),
            self.make_conv2d(in_channels=128, out_channels=128),
            torch.nn.ReLU(inplace=True),
            self.make_maxpool2d(),
            self.make_conv2d(in_channels=128, out_channels=256),
            torch.nn.ReLU(inplace=True),
            self.make_conv2d(in_channels=256, out_channels=256),
            torch.nn.ReLU(inplace=True),
            self.make_conv2d(in_channels=256, out_channels=256),
            torch.nn.ReLU(inplace=True),
            self.make_maxpool2d(),
            self.make_conv2d(in_channels=256, out_channels=512),
            torch.nn.ReLU(inplace=True),
            self.make_conv2d(in_channels=512, out_channels=512),
            torch.nn.ReLU(inplace=True),
            self.make_conv2d(in_channels=512, out_channels=512),
            torch.nn.ReLU(inplace=True),
            self.make_maxpool2d(),
            self.make_conv2d(in_channels=512, out_channels=512),
            torch.nn.ReLU(inplace=True),
            self.make_conv2d(in_channels=512, out_channels=512),
            torch.nn.ReLU(inplace=True),
            self.make_conv2d(in_channels=512, out_channels=512),
            torch.nn.ReLU(inplace=True),
            self.make_maxpool2d()
        )
        self.avgpool = torch.nn.Sequential(
            # torch.nn.AdaptiveAvgPool2d((7, 7)), # is not needed if we test with fixed sized images of 224 by 224.
            torch.nn.Flatten()
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=7*7*512, out_features=4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(in_features=4096, out_features=4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(in_features=4096, out_features=1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def make_conv2d(self, in_channels, out_channels):
        return torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def make_maxpool2d(self):
        return torch.nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )