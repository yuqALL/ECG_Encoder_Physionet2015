#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
import numpy as np


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv1d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv1d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv1d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv1d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv1d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


def initialize_weights(net_l, scale=1.0):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, dense=True, use_norm=False, drop=0.1):
        super(ResidualDenseBlock, self).__init__()
        self.dense = dense
        self.use_norm = use_norm
        self.conv1 = nn.Conv1d(nf, nf, 3, 1, 1, bias=not use_norm)
        self.conv2 = nn.Conv1d(nf, nf, 3, 1, 1, bias=not use_norm)
        self.drop = nn.Dropout(p=drop)
        self.relu = nn.LeakyReLU(0.2, True)
        if use_norm:
            self.norm_layer = nn.BatchNorm1d(nf)
            initialize_weights([self.norm_layer], 0.1)
        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_norm:
            out = self.norm_layer(out)
        out = self.relu(out)
        if self.dense:
            return identity + out
        return out


class DownBlock(nn.Module):
    def __init__(self, nf=64, output_nc=2, dense=True, use_norm=False, res_norm=False):
        super(DownBlock, self).__init__()
        self.dense = dense
        self.use_norm = use_norm
        self.conv1 = nn.Conv1d(nf, output_nc, 4, 2, 1, bias=not use_norm)
        self.conv2 = ResidualDenseBlock(output_nc, dense=self.dense, use_norm=res_norm)
        self.relu = nn.LeakyReLU(0.2, True)
        self.pool = nn.MaxPool1d(3, 2, 1)
        if use_norm:
            self.norm_layer = nn.BatchNorm1d(output_nc)
            initialize_weights([self.norm_layer], 0.1)
        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        out = self.conv1(x)
        if self.use_norm:
            out = self.norm_layer(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.pool(out)
        return out


class UpBlock(nn.Module):
    def __init__(self, nf=64, output_nc=32, use_transpose=False, dense=False, use_norm=False, res_norm=False):
        super(UpBlock, self).__init__()
        self.tp = use_transpose
        self.use_norm = use_norm
        self.conv1 = nn.Conv1d(nf, output_nc, 3, 1, 1, bias=not use_norm)
        self.conv2 = ResidualDenseBlock(output_nc, dense=dense, use_norm=res_norm)
        self.deconv = nn.ConvTranspose1d(output_nc, output_nc, 4, 2, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        if use_norm:
            self.norm_layer = nn.BatchNorm1d(output_nc)
            initialize_weights([self.norm_layer], 0.1)
        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        out = self.conv1(x)
        if self.use_norm:
            out = self.norm_layer(out)
        out = self.lrelu(out)
        if self.tp:
            out = self.deconv(out)
        else:
            out = F.interpolate(out, scale_factor=2, mode='linear',align_corners=False)
        out = self.conv2(out)
        out = F.interpolate(out, scale_factor=2, mode='linear',align_corners=False)
        return out


class UnetEncoder(nn.Module):
    def __init__(self, ngf=4, inchans=1, input_length=3072):
        super(UnetEncoder, self).__init__()

        self.conv_first = nn.Conv1d(in_channels=inchans, out_channels=ngf, kernel_size=3, stride=1, padding=1)

        down = []
        nf = ngf
        down += [DownBlock(nf, nf, dense=True, use_norm=True), nn.LeakyReLU(0.2, True)]
        for i in range(3):
            down += [DownBlock(nf, nf * 2, dense=True, use_norm=True), nn.LeakyReLU(0.2, True)]
            nf = nf * 2
        down += [DownBlock(nf, nf, dense=True, use_norm=True), nn.LeakyReLU(0.2, True)]
        self.down = nn.Sequential(*down)
        up = []
        up += [UpBlock(nf, nf, use_transpose=True, dense=True, use_norm=True, res_norm=False),
               nn.LeakyReLU(0.2, True)]
        for i in range(3):
            up += [UpBlock(nf, nf // 2, use_transpose=True, dense=True, use_norm=True, res_norm=False),
                   nn.LeakyReLU(0.2, True)]
            nf //= 2
        up += [UpBlock(nf, nf, use_transpose=True, dense=True, use_norm=True, res_norm=False),
               nn.LeakyReLU(0.2, True)]
        self.up = nn.Sequential(*up)
        self.norm_layer = nn.BatchNorm1d(64)
        initialize_weights([self.norm_layer], 0.1)
        # self.conv_last = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=kz, padding=padw)
        self.conv_last = nn.Conv1d(in_channels=ngf, out_channels=inchans, kernel_size=3, padding=1)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # initialization
        initialize_weights([self.conv_first, self.conv_last], 0.1)

    def forward(self, input):
        x = self.lrelu(self.conv_first(input))
        d = self.down(x)
        u = self.up(d)
        latent = self.conv_last(u)
        return latent

    def get_latent(self, input):
        self.eval()
        with torch.no_grad():
            x = self.lrelu(self.conv_first(input))
            return self.down(x)


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv1d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv1d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        initialize_weights(self)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out


if __name__ == "__main__":
    from torchsummary import summary

    model = UnetEncoder(inchans=1).cuda()

    summary(model, (1, 3072), device='cuda')
