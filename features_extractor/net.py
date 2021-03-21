"""Inspired from : https://github.com/hsinyilin19/ResNetVAE/blob/master/modules.py ."""

import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F


class LymphoAutoEncoder(nn.Module):
    def __init__(self):
        super(LymphoAutoEncoder, self).__init__()

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        self.resnet = EfficientNet.from_pretrained('efficientnet-b7')
        self.conv_postresnet = nn.Conv2d(2560, 1024, (1, 1), (1, 1))

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=64, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans9 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()  # y = (y1, y2, y3) \in [0 ,1]^3
        )

        # For features
        self.adaptiveavgpool = nn.AdaptiveAvgPool2d((1, 1))

    def decode(self, x):
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = self.convTrans9(x)
        x = self.convTrans10(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        return x

    def forward(self, x):
        x = self.resnet.extract_features(x)
        hidden_features = self.adaptiveavgpool(x)  # shape 2560
        x = self.conv_postresnet(x)
        hidden_features_reduced = self.adaptiveavgpool(x)  # shape 1024
        x_reconst = self.decode(x)
        return x_reconst, hidden_features_reduced, hidden_features