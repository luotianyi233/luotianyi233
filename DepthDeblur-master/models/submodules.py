#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast

def conv(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                  padding=((kernel_size - 1) // 2) * dilation, bias=bias),
        nn.LeakyReLU(0.1, inplace=True)
    )


def upconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


def resnet_block(in_channels, kernel_size=3, dilation=[1, 1], bias=True):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0],
                      padding=((kernel_size - 1) // 2) * dilation[0], bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1],
                      padding=((kernel_size - 1) // 2) * dilation[1], bias=bias),
        )
    @autocast()
    def forward(self, x):
        out = self.stem(x) + x
        return out


def kernel2d_conv(feat_in, kernel, ksize):
    """
    If you have some problems in installing the CUDA FAC layer, 
    you can consider replacing it with this Python implementation.
    Thanks @AIWalker-Happy for his implementation.
    """
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad = (ksize - 1) // 2

    feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(N, H, W, channels, -1)

    kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize)
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)
    feat_out = torch.sum(feat_in * kernel, -1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
    return feat_out


def init_upconv_bilinear(weight):
    f_shape = weight.size()
    heigh, width = f_shape[-2], f_shape[-1]
    f = np.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([heigh, width])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weight.data.fill_(0.)
    for i in range(f_shape[0]):
        for j in range(f_shape[1]):
            weight.data[i, j, :, :] = torch.from_numpy(bilinear)


def cat_with_crop(target, input):
    output = []
    for item in input:
        if item.size()[2:] == target.size()[2:]:
            output.append(item)
        else:
            output.append(item[:, :, :target.size(2), :target.size(3)])
    output = torch.cat(output, 1)
    return output


def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad

    return hook

def ms_dilate_block(in_channels, kernel_size=3, dilation=[1,1,1,1], bias=True):
    return MSDilateBlock(in_channels, kernel_size, dilation, bias)

class MSDilateBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(MSDilateBlock, self).__init__()
        self.conv1 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[0], bias=bias)
        self.conv2 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[1], bias=bias)
        self.conv3 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[2], bias=bias)
        self.conv4 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[3], bias=bias)
        self.convi =  nn.Conv2d(in_channels*4, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=bias)
    @autocast()
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        cat  = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.convi(cat) + x
        return out