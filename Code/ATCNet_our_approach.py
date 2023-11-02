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
        in_chans=22,
        n_windows=1,  # TODO: doesn't depend on window length and input length does
        eegn_F1=16,
        eegn_kernelSize=64,
        eegn_poolSize=8,
        eegn_dropout=0.3,
        tcn_kernelSize=4,
        tcn_filters=32,  # TODO: doesn't depend on window length and input length does
        tcn_dropout=0.3,
        tcn_activation="relu",
        fuse="average",
    ):
        super(ATCNet, self).__init__()
        self.conv_block = ConvBlock(
            in_chans, eegn_F1, eegn_kernelSize, eegn_poolSize, dropout=eegn_dropout
        )
        self.tcn_block = TCNBlock(
            eegn_F1, 2, tcn_kernelSize, dropout=tcn_dropout, filters=tcn_filters
        )
        self.n_windows = n_windows
        self.fuse = fuse

    def forward(self, x):
        x = self.conv_block(x)
        x = x[:, :, -1, :]  # Select last channel of last time step

        sw_concat = []
        for st in range(self.n_windows):
            end = x.shape[1] - self.n_windows + st + 1

            block = x[:, st:end, :]
            block = self.tcn_block(block)
            block = block[:, -1, :]

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
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F2,
            kernel_size=(1, in_chans),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu = nn.ELU()
        self.avg_pool = nn.AvgPool2d(
            kernel_size=(poolSize, 1), stride=(1, 1)
        )  # TODO: figure out how stride is translated, default was the same as kernel
        self.dropout = nn.Dropout(dropout)

        # Block 3
        self.conv2 = nn.Conv2d(F2, F2, kernel_size=(16, 1), padding="same")
        self.bn3 = nn.BatchNorm2d(F2)
        self.avg_pool2 = nn.AvgPool2d(
            kernel_size=(poolSize, 1), stride=(1, 1)
        )  # TODO: figure out how stride is translated, default was the same as kernel
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
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
    def __init__(self, input_layer, depth, kernel_size, filters, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.layers = []
        # Block 1
        self.conv1 = nn.Conv1d(
            input_layer,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            dilation=1,
            bias=True,
        )
        self.bn1 = nn.BatchNorm1d(filters)
        self.linear = nn.Linear(filters, filters)
        self.dropout = nn.Dropout(dropout)

        # Block 2
        self.conv2 = nn.Conv1d(
            filters,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            dilation=1,
            bias=True,
        )
        self.bn2 = nn.BatchNorm1d(filters)
        self.linear = nn.Linear(filters, filters)
        self.dropout = nn.Dropout(dropout)

        for i in range(depth):
            dilation_size = 2 ** (i + 1)
            self.layers += [
                nn.Conv1d(
                    filters,
                    filters,
                    kernel_size=kernel_size,
                    dilation=dilation_size,
                    padding=dilation_size,
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
                    padding=dilation_size,
                    bias=False,
                ),
                nn.BatchNorm1d(filters),
                nn.Dropout(dropout),
                nn.ReLU(),
            ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
