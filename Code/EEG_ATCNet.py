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


class ConvBlock(nn.Module):
    def __init__(self, input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.1):
        super(ConvBlock, self).__init__()
        F2 = F1 * D
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=F1, out_channels=F1, kernel_size=(kernLength, 1), 
                               padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2
        self.depthwise_conv = nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=(1, in_chans),
                                        groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu = nn.ELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(poolSize, 1))
        self.dropout = nn.Dropout(dropout)

        # Block 3
        self.conv2 = nn.Conv2d(F2, F2, kernel_size=(16, 1), padding='same')
        self.bn3 = nn.BatchNorm2d(F2)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(poolSize, 1))
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
    def __init__(self, input_dimension, depth, kernel_size, filters, dropout):
        super(TCNBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv1d(input_dimension, filters, kernel_size=1, padding=0))
        self.layers.append(nn.BatchNorm1d(filters))
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.ReLU())

        for i in range(depth):
            dilation_size = 2 ** (i + 1)
            self.layers.append(
                nn.Conv1d(filters, filters, kernel_size=kernel_size, dilation=dilation_size, padding=dilation_size,
                          bias=False))
            self.layers.append(nn.BatchNorm1d(filters))
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ATCNet(nn.Module):
    def __init__(self, in_chans=22, n_windows=3,
                 eegn_F1=16, eegn_kernelSize=64, eegn_poolSize=8, eegn_dropout=0.3,
                 tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
                 tcn_activation='relu', fuse='average'):
        super(ATCNet, self).__init__()
        self.conv_block = ConvBlock(in_chans, eegn_F1, eegn_kernelSize, eegn_poolSize, dropout=eegn_dropout)
        self.tcn_block = TCNBlock(eegn_F1, tcn_filters, tcn_kernelSize, tcn_dropout, tcn_activation)
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

        if self.fuse == 'average':
            sw_concat = torch.mean(torch.stack(sw_concat), dim=0)
        elif self.fuse == 'concat':
            sw_concat = torch.cat(sw_concat, dim=1)

        return F.softmax(sw_concat, dim=1)
