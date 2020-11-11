#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2020-11-05 20:48:27
LastEditTime: 2020-11-11 17:18:43
Description: PNN (SRCMM-based), Pansharpening by Convolutional Neural Networks, 1e6, batch_size = 128, learning_rate = 1e-4
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

        base_filter = 64
        num_channels = 5
        out_channels = 4
        self.head = ConvBlock(num_channels, 64, 9, 1, 4, activation='relu', norm=None, bias = True)

        self.body = ConvBlock(base_filter, 32, 5, 1, 2, activation='relu', norm=None, bias = True)

        self.output_conv = ConvBlock(32, out_channels, 5, 1, 2, activation='relu', norm=None, bias = True)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    # torch.nn.init.kaiming_normal_(m.weight)
        	    torch.nn.init.xavier_uniform_(m.weight, gain=1)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    # torch.nn.init.kaiming_normal_(m.weight)
        	    torch.nn.init.xavier_uniform_(m.weight, gain=1)
        	    if m.bias is not None:
        		    m.bias.data.zero_()

    def forward(self, x_ms, x_pan):

        x_f = torch.cat((x_ms, x_pan), 1)
        x_f = self.head(x_f)
        x_f = self.body(x_f)
        x_f = self.output_conv(x_f)

        return x_f