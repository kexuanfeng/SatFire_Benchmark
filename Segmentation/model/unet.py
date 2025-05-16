import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import os
import numpy as np


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet, self).__init__()
        self.encoder1 = self.contracting_block(in_channels, 64)
        self.encoder2 = self.contracting_block(64, 128)
        self.encoder3 = self.contracting_block(128, 256)
        self.encoder4 = self.contracting_block(256, 512)

        self.middle = self.middle_diy(512, 512)

        self.decoder4 = self.expansive_block(1024, 512)
        self.decoder3 = self.expansive_block(512, 256)
        self.decoder2 = self.expansive_block(256, 128)
        self.decoder1 = self.expansive_block(128, 64)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

    def middle_diy(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),

        )
        return block

    def expansive_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)
        )
        return block

    def forward(self, x):
        # Encoding
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Middle
        middle = self.middle(enc4)

        # Decoding + Concatenation
        # print(middle.shape,enc4.shape)
        dec4 = self.decoder4(torch.cat([middle, enc4], 1))
        dec3 = self.decoder3(torch.cat([dec4, enc3], 1))
        dec2 = self.decoder2(torch.cat([dec3, enc2], 1))
        dec1 = self.decoder1(torch.cat([dec2, enc1], 1))

        out = self.final(dec1)

        return torch.sigmoid(out)

