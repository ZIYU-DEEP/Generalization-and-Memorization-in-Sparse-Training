"""
Title: mnist_alexnet.py
Description: The file for alexnet of mnist.
Warning: not test the file yet!! (Feb 13, 2022)
"""
from .base_net import BaseNet
import torch.nn as nn


class MNISTLeNet(BaseNet):
    def __init__(self):
        super(MNISTLeNet, self).__init__()
        self.cnn_model = nn.Sequential(         # nn.Sequentila allows multiple layers to stack together
            nn.Conv2d(1,6,5),                   #(N,1,28,28) -> (N,6,24,24)
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),           #(N,6,24,24) -> (N,6,12,12)
            nn.Conv2d(6, 16, 5),                  #(N,6,12,12) -> (N,16,8,8)
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2)            #(N,16,8,8) -> (N,16,4,4)
            )

        self.fc_model = nn.Sequential(          # Fully connected layer
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10))


    def forward(self,x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), - 1)    # Flatning the inputs from tensors to vectors
        x = self.fc_model(x)        # Passing the conv layer to fully connected layer
        return x
