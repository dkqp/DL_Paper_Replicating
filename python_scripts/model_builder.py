'''
Contains Pytorch model code to instantiate a TinyVGG model from the CNN Explainer website.
'''

import torch
from torch import nn

class TinyVGG(nn.Module):
    ''' TinyVGG class
    Args:
        input_shape: num of channels of input image
        hidden_units: num of hidden units
        output_shape: num of output classes

    Returns:
        TinyVGG classifier module
    '''
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13, out_features=output_shape)
        )

    def forward(self, x):
        return self.classifier_layer(self.conv_block_2(self.conv_block_1(x)))

class PatchEmbedding(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            patch_size: int = 16,
            embedding_dim: int = 768
    ) -> None:
        super().__init__()

        self.patch_size = patch_size

        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )

        self.flatten = nn.Flatten(
            start_dim=2,
            end_dim=3
        )

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f'Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {self.patch_size}'

        return self.flatten(self.patcher(x)).permute(0, 2, 1)

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 768,
            num_heads: int = 12,
            attn_dropout: int = 0
    ) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True # (batch, seq(number_or_patches), feature(embedding_dimension))
        )

    def forward(self, x):
        ln_out = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=ln_out, key=ln_out, value=ln_out, need_weights=False)
        return attn_output + x

class MultiLayerPerceptronBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 768,
            mlp_size: int = 3072,
            dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp_layer = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        ln_out = self.layer_norm(x)
        mlp_out = self.mlp_layer(ln_out)
        return mlp_out + x

class TransformerEncoderBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 768,
            num_heads: int = 12,
            mlp_size: int = 3072,
            mlp_dropout: float = 0.1,
            attn_dropout: float = 0
    ) -> None:
        super().__init__()

        self.msa_block = MultiHeadSelfAttentionBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout
        )

        self.mlp_block = MultiLayerPerceptronBlock(
            embedding_dim=embedding_dim,
            mlp_size=mlp_size,
            dropout=mlp_dropout
        )

    def forward(self, x):
        return self.mlp_block(self.msa_block(x))

class ViT(nn.Module):
    def __init__(
            self,
            img_size: int = 224,
            in_channels: int = 3,
            patch_size: int = 16,
            num_transformer_layers: int = 12,
            embedding_dim: int = 768,
            mlp_size: int = 3072,
            num_heads: int = 12,
            attn_dropout: float = 0,
            mlp_dropout: float = 0.1,
            embedding_dropout: float = 0.1,
            num_classes: int = 1000
    ) -> None:
        super().__init__()

        assert img_size % patch_size == 0, f'Image size must be divisible by patch size, image: {img_size}, patch: {patch_size}'

        self.num_patches = img_size ** 2 // patch_size ** 2

        self.class_embedding = nn.Parameter(
            data=torch.randn(1, 1, embedding_dim),
            requires_grad=True
        )

        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches + 1, embedding_dim))

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embedding_dim=embedding_dim
        )

        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                mlp_dropout=mlp_dropout,
                attn_dropout=attn_dropout,
            ) for _ in range(num_transformer_layers)]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        patched_x = self.patch_embedding(x)
        patched_and_positioned_x = torch.concat((self.class_embedding.expand(batch_size, -1, -1), patched_x), dim=1) + self.position_embedding
        embedding_dropped_out_x = self.embedding_dropout(patched_and_positioned_x)
        transformed_x = self.transformer(embedding_dropped_out_x)
        return self.classifier(transformed_x[:, 0, :])

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
