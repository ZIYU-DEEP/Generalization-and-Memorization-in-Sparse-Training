"""
Title: mlp.py
Description: The file for a fully connected network.
"""


from .base_net import BaseNet
from helper import utils
import torch.nn as nn


class MLP(BaseNet):
    def __init__(self,
                 in_dim: int=12,
                 out_dim: int=2,
                 hidden_act: str='tanh',
                 out_act: str='softmax',
                 hidden_dims: str='10-7-5-4-3'):
        super(MLP, self).__init__()

        # Initialize parameters
        hidden_act = utils.act_dict(hidden_act)  # add helper here
        out_act = utils.act_dict(out_act)  # add helper here
        neurons = [in_dim, *utils.str_to_list(hidden_dims)]  # add helper here

        # Create input and hidden layers
        net = nn.Sequential()
        for i in range(len(neurons) - 1):
            net.add_module(f'dense_{i}', nn.Linear(neurons[i], neurons[i + 1]))
            net.add_module(f'act_{i}', hidden_act)

        # Create the output layer
        net.add_module('dense_out', nn.Linear(neurons[-1], out_dim))
        # net.add_module('act_out', out_act)  # nn.CrossEntropy will add it

        self.net = net

    def forward(self, x):
        x = x.view(x.size(0), -1)

        for layer in self.net:
            x = layer(x)
        return x
