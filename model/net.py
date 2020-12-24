import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.measure import label, regionprops
import pdb


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    print(net)
    print('Total number of parameters: %d' % num_params)


class SELayer(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Sequential(
                nn.Conv3d(channel, channel//reduction, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, channel, kernel_size=1, stride=1),
                nn.Sigmoid()
                )
    def forward(self, x):

        y = self.avg_pool(x)
        y = self.conv(y)

        return x * y

def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation_rate=1):
    if kernel_size == (1,3,3):
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, \
                padding=(0,1,1), bias=False, dilation=dilation_rate)

    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,\
                padding=padding, bias=False, dilation=dilation_rate)

class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, reduction=4, dilation_rate=1, norm='bn'):
        super(SEBasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, kernel_size=kernel_size, stride=stride)
        if norm == 'bn':
            self.bn1 = nn.BatchNorm3d(inplanes)
        elif norm =='in':
            self.bn1 = nn.InstanceNorm3d(inplanes)
        elif norm =='gn':
            self.bn1 = nn.GroupNorm(NUM_GROUP, inplanes)
        else:
            raise ValueError('unsupport norm method')
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=dilation_rate)
        if norm == 'bn':
            self.bn2 = nn.BatchNorm3d(planes)
        elif norm =='in':
            self.bn2 = nn.InstanceNorm3d(planes)
        elif norm =='gn':
            self.bn2 = nn.GroupNorm(NUM_GROUP, planes)
        else:
            raise ValueError('unsupport norm method')
        self.se = SELayer(planes, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            if norm == 'bn':
                self.shortcut = nn.Sequential(
                    nn.BatchNorm3d(inplanes),
                    self.relu,
                    nn.Conv3d(inplanes, planes, kernel_size=1, \
                            stride=stride, bias=False)
                )
            elif norm =='in':
                self.shortcut = nn.Sequential(
                    nn.InstanceNorm3d(inplanes),
                    self.relu,
                    nn.Conv3d(inplanes, planes, kernel_size=1, \
                            stride=stride, bias=False)
                )
            elif norm =='gn':
                self.shortcut = nn.Sequential(
                    nn.GroupNorm(NUM_GROUP, inplanes),
                    self.relu,
                    nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
                )
            else:
                raise ValueError('unsupport norm method')

        self.stride = stride

    def forward(self, x):
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.se(out)

        out += self.shortcut(residue)

        return out

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, se=False, norm='bn'):
        super(inconv, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = SEBasicBlock(out_ch, out_ch, kernel_size=(1,3,3), norm=norm)

    def forward(self, x): 

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        return out 




class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, se=False, reduction=2, dilation_rate=1, norm='bn'):
        super(conv_block, self).__init__()

        self.conv = SEBasicBlock(in_ch, out_ch, stride=stride, reduction=reduction, dilation_rate=dilation_rate, norm=norm)

    def forward(self, x):

        out = self.conv(x)

        return out

class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, scale=(2,2,2), se=False, reduction=2, norm='bn'):
        super(up_block, self).__init__()

        self.scale = scale

        self.conv = nn.Sequential(
            conv_block(in_ch+out_ch, out_ch, se=se, reduction=reduction, norm=norm),
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=self.scale, mode='trilinear', align_corners=True)

        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)

        return out

class up_nocat(nn.Module):
    def __init__(self, in_ch, out_ch, scale=(2,2,2), se=False, reduction=2, norm='bn'):
        super(up_nocat, self).__init__()

        self.scale = scale
        self.conv = nn.Sequential(
            conv_block(out_ch, out_ch, se=se, reduction=reduction, norm=norm),
        )

    def forward(self, x):
        out = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=True)
        out = self.conv(out)

        return out

class literal_conv(nn.Module):
    def __init__(self, in_ch, out_ch, se=False, reduction=2, norm='bn'):
        super(literal_conv, self).__init__()

        self.conv = conv_block(in_ch, out_ch, se=se, reduction=reduction, norm=norm)

    def forward(self, x):

        out = self.conv(x)

        return out
class DenseASPPBlock(nn.Sequential):
    """Conv Net block for building DenseASPP"""

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True, norm='bn'):
        super(DenseASPPBlock, self).__init__()
        if bn_start:
            if norm == 'bn':
                self.add_module('norm_1', nn.BatchNorm3d(input_num))
            elif norm == 'in':
                self.add_module('norm_1', nn.InstanceNorm3d(input_num))
            elif norm == 'gn':
                self.add_module('norm_1', nn.GroupNorm(NUM_GROUP, input_num))

        self.add_module('relu_1', nn.ReLU(inplace=True))
        self.add_module('conv_1', nn.Conv3d(in_channels=input_num, out_channels=num1, kernel_size=1))

        if norm == 'bn':
            self.add_module('norm_2', nn.BatchNorm3d(num1))
        elif norm == 'in':
            self.add_module('norm_2', nn.InstanceNorm3d(num1))
        elif norm == 'gn':
            self.add_module('norm_2', nn.GroupNorm(NUM_GROUP, num1))
        self.add_module('relu_2', nn.ReLU(inplace=True))
        self.add_module('conv_2', nn.Conv3d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate))

        self.drop_rate = drop_out

    def forward(self, input):
        feature = super(DenseASPPBlock, self).forward(input)

        if self.drop_rate > 0:
            feature = F.dropout3d(feature, p=self.drop_rate, training=self.training)

        return feature


