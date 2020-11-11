#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2020-11-05 20:47:04
LastEditTime: 2020-11-11 16:29:08
Description: PanNet: A deep network architecture for pan-sharpening
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net, self).__init__()

        self.trans = TransConvBlock(4, 4, kernel_size=8, stride=4, padding=1)

        base_filter = 32
        body = [
            ResnetBlock(base_filter, 3, 1, 1, 0.1, activation='relu', norm=None) for _ in range(n_resblocks)
        ]
        self.body = nn.Sequential(*body)

        self.output_conv = ConvBlock(base_filter, 4, 3, 1, 1, activation='relu', norm=None)
    
    def forward(self, x_ms, x_pan):
        x_f = torch.cat((x_ms_up, x_pan), 1)
        x_f = self.body(x_f)
        x_f = self.output_conv(x_f)
        return x_f
        