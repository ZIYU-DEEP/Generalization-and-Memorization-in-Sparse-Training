"""
Title: mnist_alexnet.py
Description: The file for alexnet of mnist.
"""
from .base_net import BaseNet
import torch.nn as nn


class MNISTAlexNet(BaseNet):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # 32 * 28 * 28
            nn.MaxPool2d(kernel_size=2, stride=2), # 32 * 14 * 14
            nn.ReLU(inplace=True),
            )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64 * 14 * 14
            nn.MaxPool2d(kernel_size=2, stride=2), # 64 * 7 * 7
            nn.ReLU(inplace=True),
            )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128 * 7 * 7
            )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 256 * 7 * 7
            )

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 256 * 7 * 7
            nn.MaxPool2d(kernel_size=3, stride=2), # 256 * 3 * 3
            nn.ReLU(inplace=True),
            )

        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
