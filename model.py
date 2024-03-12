import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# This model is inspired by PANN's CNN Architectures:

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, 3), stride=(1, 1),
                               padding=(0, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(1, 3), stride=(1, 1),
                               padding=(0, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class CRNN(nn.Module):
    def __init__(self, classes_num):

        super(CRNN, self).__init__()

        gru_input_size = 64 * 5 #gru_input_size = cnn_output_height * 64
        gru_hidden_size = 32
        gru_num_layers = 2

        self.bn0 = nn.BatchNorm2d(40)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=16)
        self.conv_block2 = ConvBlock(in_channels=16, out_channels=32)
        self.conv_block3 = ConvBlock(in_channels=32, out_channels=64)

        self.gru = nn.GRU(gru_input_size, gru_hidden_size, gru_num_layers,
                          batch_first=True, bidirectional=True)

        self.fc = nn.Linear(gru_hidden_size*2, classes_num)

    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        x = torch.unsqueeze(input, dim=1)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = self.conv_block1(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # batch, channel, frame, mel = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.gru(x)
        x = self.fc(x)
        return x