"""
This file is based on the model architecture proposed in 
"Physics-informed attention temporal convolutional network for EEG-based motor imagery classification"
Authors: Hamdi Altaheri, Ghulam Muhammad, Mansour Alsulaiman
Center of Smart Robotics Research, King Saud University, Saudi Arabia
https://doi.org/10.1109/TII.2022.3197419

and their GitHub repository: https://github.com/Altaheri/EEG-ATCNet
however our approach is based on PyTorch not on TensorFlow.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ATCNet(nn.Module):
    def __init__(
        self,
        in_samples=1125,
        in_chans=22,
        n_classes=4,
        n_windows=5,
        eegn_F1=16,
        eegn_D=2,
        eegn_kernelSize=64,
        eegn_poolSize=8,
        eegn_dropout=0.3,
        tcn_kernelSize=5,  # TODO: changed to an odd number
        tcn_filters=32,
        tcn_depth=2,
        tcn_dropout=0.3,
        tcn_activation="relu",
        fuse="average",
        attention="mha",
    ):
        super(ATCNet, self).__init__()
        self.conv_block = ConvBlock(
            input_layer=in_chans,
            F1=eegn_F1,
            kernLength=eegn_kernelSize,
            poolSize=eegn_poolSize,
            D=eegn_D,
            dropout=eegn_dropout,
        )
        self.tcn_block = TCNBlock(  # TODO: doubt about input layer here
            input_layer=tcn_filters,
            input_dimension=eegn_F1 * eegn_D,
            depth=tcn_depth,
            kernel_size=tcn_kernelSize,
            dropout=tcn_dropout,
            filters=tcn_filters,
        )
        # self.attention_block = MhaBlock()
        self.n_windows = n_windows
        self.dense = nn.Linear(tcn_filters, n_classes)
        self.fuse = fuse

    def forward(self, x):
        x = self.conv_block(x)
        x = x[:, :, -1, :]  # Select last channel of last time step

        sw_concat = []
        for st in range(self.n_windows):
            end = x.shape[2] - self.n_windows + st + 1

            block = x[:, :, st:end]

            if hasattr(self, "attention"):
                block = self.attention_block(block)
            block = self.tcn_block(block)
            block = block[:, :, -1]
            block = self.dense(block)
            sw_concat.append(block)

        if self.fuse == "average":
            sw_concat = torch.mean(torch.stack(sw_concat), dim=0)
        elif self.fuse == "concat":
            sw_concat = torch.cat(sw_concat, dim=1)

        return F.softmax(sw_concat, dim=1)


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_layer,
        F1=4,
        kernLength=64,
        poolSize=8,
        D=2,
        in_chans=22,
        dropout=0.1,
    ):
        super(ConvBlock, self).__init__()
        F2 = F1 * D
        # Block 1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(kernLength, 1),
            padding="same",
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F2,
            kernel_size=(in_chans, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu = nn.ELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout = nn.Dropout(dropout)

        # Block 3
        self.conv2 = nn.Conv2d(F2, F2, kernel_size=(16, 1), padding="same", bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, poolSize))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.tensor(x).float()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        return x


class TCNBlock(nn.Module):
    def __init__(
        self, input_layer, input_dimension, depth, kernel_size, filters, dropout=0.2
    ):
        super(TCNBlock, self).__init__()
        self.layers1 = [
            nn.Conv1d(
                input_layer,
                out_channels=filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                dilation=1,
                bias=True,
            ),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                filters,
                out_channels=filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                dilation=1,
                bias=True,
            ),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

        self.mid = (
            nn.Conv1d(filters, filters, kernel_size=1)
            if input_dimension != filters
            else lambda x: x
        )

        self.layers2 = [nn.ReLU()]
        for i in range(depth - 1):
            dilation_size = 2 ** (i + 1)
            self.layers2 += [
                nn.Conv1d(
                    filters,
                    filters,
                    kernel_size=kernel_size,
                    dilation=dilation_size,
                    padding=dilation_size + kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(filters),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Conv1d(
                    filters,
                    filters,
                    kernel_size=kernel_size,
                    dilation=dilation_size,
                    padding=dilation_size + kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(filters),
                nn.Dropout(dropout),
                nn.ReLU(),
            ]

    def forward(self, x):
        input_layer = x
        for layer in self.layers1:
            x = layer(x)

        input_layer = self.mid(input_layer)
        x += input_layer

        for layer in self.layers2:
            x = layer(x)
        return x


class MhaBlock(nn.Module):
    def __init__(
        self, normalized_shape=(2, 32, 16), epsilon=1e-6, num_heads=2, dropout=0.5
    ):
        super().__init__()
        self.normalization = nn.LayerNorm(
            normalized_shape=normalized_shape, eps=epsilon
        )
        self.mha = nn.MultiheadAttention(
            embed_dim=normalized_shape[-1], num_heads=num_heads, dropout=dropout
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        initial = x
        x = self.normalization(x)
        x = self.mha(x, x, x)
        x = self.dropout(x)
        return x + initial
