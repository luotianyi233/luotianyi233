#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by orz

from models.submodules import *
from torch.cuda.amp import autocast as autocast
class Encoder(nn.Module):
    def __init__(self, in_channel=3):
        super(Encoder, self).__init__()
        # encoder
        ks = 3
        self.conv1_1 = conv(in_channel, 32, kernel_size=ks, stride=1)
        self.conv1_2 = resnet_block(32, kernel_size=ks)
        self.conv1_3 = resnet_block(32, kernel_size=ks)
        self.conv1_4 = resnet_block(32, kernel_size=ks)

        self.conv2_1 = conv(32, 64, kernel_size=ks, stride=2)
        self.conv2_2 = resnet_block(64, kernel_size=ks)
        self.conv2_3 = resnet_block(64, kernel_size=ks)
        self.conv2_4 = resnet_block(64, kernel_size=ks)

        self.conv3_1 = conv(64, 128, kernel_size=ks, stride=2)
        self.conv3_2 = resnet_block(128, kernel_size=ks)
        self.conv3_3 = resnet_block(128, kernel_size=ks)
        self.conv3_4 = resnet_block(128, kernel_size=ks)

        dilation = [1,2,3,4]
        self.convd_1 = resnet_block(128, kernel_size=ks, dilation = [2, 1])
        self.convd_2 = resnet_block(128, kernel_size=ks, dilation = [3, 1])
        self.convd_3 = ms_dilate_block(128, kernel_size=ks, dilation = dilation)

    @autocast()
    def forward(self, img):
        # encoder
        conv1 = self.conv1_4(self.conv1_3(self.conv1_2(self.conv1_1(img))))
        conv2 = self.conv2_4(self.conv2_3(self.conv2_2(self.conv2_1(conv1))))
        conv3 = self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(conv2))))
        convd = self.convd_3(self.convd_2(self.convd_1(conv3)))

        return convd, conv3, conv2, conv1

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        ks = 3

        # decoder
        self.upconv3_i = conv(256, 128, kernel_size=ks,stride=1)
        self.upconv3_3 = resnet_block(128, kernel_size=ks)
        self.upconv3_2 = resnet_block(128, kernel_size=ks)
        self.upconv3_1 = resnet_block(128, kernel_size=ks)

        self.upconv2_u = upconv(128, 64)
        self.upconv2_i = conv(192, 64, kernel_size=ks,stride=1)
        self.upconv2_3 = resnet_block(64, kernel_size=ks)
        self.upconv2_2 = resnet_block(64, kernel_size=ks)
        self.upconv2_1 = resnet_block(64, kernel_size=ks)

        self.upconv1_u = upconv(64, 32)
        self.upconv1_i = conv(96, 32, kernel_size=ks,stride=1)
        self.upconv1_3 = resnet_block(32, kernel_size=ks)
        self.upconv1_2 = resnet_block(32, kernel_size=ks)
        self.upconv1_1 = resnet_block(32, kernel_size=ks)

        self.img_prd = conv(32, 3, kernel_size=ks, stride=1)

    @autocast()
    def forward(self, convd, conv3, conv2, conv1):
        # decoder
        cat3 = self.upconv3_i(convd)
        upconv2 = self.upconv2_u(self.upconv3_1(self.upconv3_2(self.upconv3_3(cat3))))
        cat2 = self.upconv2_i(cat_with_crop(conv2, [conv2, upconv2]))
        upconv1 = self.upconv1_u(self.upconv2_1(self.upconv2_2(self.upconv2_3(cat2))))
        cat1 = self.upconv1_i(cat_with_crop(conv1, [conv1, upconv1]))
        img_res = self.img_prd(self.upconv1_1(self.upconv1_2(self.upconv1_3(cat1))))

        return img_res


class DepthDeblurDAVANet(nn.Module):
    def __init__(self):
        super(DepthDeblurDAVANet, self).__init__()
        self.encoder_rgb = Encoder(3)
        self.encoder_depth = Encoder(1)
        self.decoder = Decoder()

    @autocast()
    def forward(self, rgb, depth):
        # encoder
        conv_rgb = self.encoder_rgb(rgb)
        conv_depth = self.encoder_depth(depth)
        convd = torch.cat([conv_rgb[0], conv_depth[0]], 1)
        conv3 = torch.cat([conv_rgb[1], conv_depth[1]], 1)
        conv2 = torch.cat([conv_rgb[2], conv_depth[2]], 1)
        conv1 = torch.cat([conv_rgb[3], conv_depth[3]], 1)

        # decoder
        img_prd = self.decoder(convd, conv3, conv2, conv1)
        img_prd = img_prd + rgb

        return img_prd