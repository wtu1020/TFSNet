import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import numpy as np


class TemporalRelayBlock(nn.Module):
    def __init__(self, channel, nhead, num_layers, dim_feedforward=16, dropout=0.):
        self.channel = channel
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        super(TemporalRelayBlock, self).__init__()

        # Transformer encoder layers
        self.encoder_layer = TransformerEncoderLayer(self.channel, self.nhead, self.dim_feedforward, self.dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, self.num_layers)

        # LSTM layers
        self.lstm = nn.LSTM(input_size=self.channel, hidden_size=self.channel,  num_layers=1, dropout=self.dropout, batch_first=True)

        # Batch normalization and GeLU activation
        self.bn = nn.BatchNorm1d(self.channel)
        self.gelu = nn.GELU()

        self.max_pool = nn.MaxPool1d(kernel_size=5, stride=2, padding=1)
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)

        x = self.bn(x)
        x = self.gelu(x)
        x = self.max_pool(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding='same', groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ParallelFusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=None):
        super(ParallelFusionConv, self).__init__()

        if kernel_size is None:
            kernel_size = [3, 6]
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size[0], stride=1)
        self.conv2 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size[1], stride=1)
        self.bn = nn.BatchNorm1d(out_channels*2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x_con = torch.cat([x1, x2], dim=1)
        out = self.bn(x_con)
        out = self.relu(out)
        return out


class Attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


def dct_filters(k=3, groups=1, expand_dim=1, level=None):

    num_filters = k if level is None else level
    filter_bank = np.zeros((num_filters, k), dtype=np.float32)
    filter_index = 0

    for i in range(k):
        if level is not None and i >= level:
            continue
        filter_values = [
            math.cos((math.pi * (x + 0.5) * i) / k) for x in range(k)
        ]
        filter_bank[filter_index, :] = filter_values
        filter_bank[filter_index, :] /= np.sum(np.abs(filter_bank[filter_index, :]))
        filter_index += 1

    filter_bank = np.expand_dims(filter_bank, axis=expand_dim)
    filter_bank = np.tile(filter_bank, (groups, 1, 1))
    return torch.FloatTensor(filter_bank)


class DiscreteSpectralConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation, bias=True, level=None, groups=1, dropout=0.):
        super(DiscreteSpectralConv, self).__init__()

        # Setup parameters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dropout = dropout

        # Initialize DCT Filters
        self.dct = nn.Parameter(
            dct_filters(
                k=kernel_size,
                groups=1,
                expand_dim=0,
                level=level
            ),
            requires_grad=False
        )
        num_filters = self.dct.shape[1]
        weight_shape = (out_channel, in_channel // groups, num_filters, 1)

        self.weight = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(*weight_shape), mode='fan_out', nonlinearity='relu'))
        self.bias = nn.Parameter(nn.init.zeros_(torch.Tensor(out_channel))) if bias else None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x):
        filt = torch.sum(self.weight * self.dct, dim=2)
        x = F.conv1d(x, filt, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, in_channels, channels, num_classes):
        super(Model, self).__init__()
        self.in_channel = in_channels
        self.num_classes = num_classes
        self.channel = channels

        self.PFblock1 = ParallelFusionConv(self.in_channel, self.channel)
        self.PFblock2 = ParallelFusionConv(self.channel * 2, self.channel * 2)
        self.TRBlock = TemporalRelayBlock(channel=self.channel*2, nhead=2, num_layers=1, dropout=0.1)

        self.DSconv1 = DiscreteSpectralConv(in_channel=self.in_channel, out_channel=self.channel * 2, kernel_size=3, stride=2, padding=1, dilation=2, dropout=0.1)
        self.DSconv2 = DiscreteSpectralConv(in_channel=self.channel * 2, out_channel=self.channel * 4, kernel_size=3, stride=2, padding=1, dilation=2, dropout=0.1)

        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(self.channel * 8, num_classes)
        self.Att = Attention(channel=self.channel * 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.PFblock1(x)
        x1 = self.max_pool(x1)
        x1 = self.TRBlock(x1)
        x1 = self.PFblock2(x1)
        x1 = self.adaptive_avg_pool(x1)

        x2 = self.DSconv1(x)
        x2 = self.max_pool(x2)
        x2 = self.Att(x2)
        x2 = self.DSconv2(x2)
        x2 = self.adaptive_avg_pool(x2)

        out = torch.cat((x1, x2), dim=1)
        out = torch.flatten(out, 1)
        out_log = self.fc(out)
        output = self.softmax(out_log)
        return out_log, output