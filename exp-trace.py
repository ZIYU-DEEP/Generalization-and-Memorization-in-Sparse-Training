"""
[Title] experiment-trace.py
[Usage] This is a file to calculate the hessian traces.
"""

# from pyhessian import hessian
import pyhessian
from helper import utils, pruner
from pathlib import Path
from torch import nn
from PIL import Image
from functools import reduce
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from loader.loader_cifar10 import CIFAR10Loader
from loader.loader_cifar100 import CIFAR100Loader
from network.res_net import ResNet
from collections import OrderedDict

import math
import time
import torch
import joblib
import logging
import torch.nn
import argparse
import numpy as np
import seaborn as sea
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.utils.prune as torch_prune
import torchvision.transforms as transforms


# ##################################################################
# 0. Set the arguments
# ##################################################################
parser = argparse.ArgumentParser()

parser.add_argument('-pt', '--path', type=str, default='./final_path',
                    help='The path to get results.')
parser.add_argument('-pi', '--prune_indicator', type=int, default=1,
                    help='1 if prune, else 0.')
parser.add_argument('-pr', '--prune_ratio', type=float, default=0.9,
                    help='The ratio for sparse training')

# Better to leave them as default
parser.add_argument('-ul', '--use_loader', type=str, default='train',
                    help='The loader to use in evaluating the fisher.',
                    choices=['train', 'test', 'clean', 'noisy'])
parser.add_argument('-dv', '--device', type=str, default='cuda',
                    help='Choose from cpu, cuda, and tpu.')
parser.add_argument('-bs', '--batch_size', type=int, default=512,
                    help='The batch size for training.')

p = parser.parse_args()


# ##################################################################
# 0. Define Global Variables
# ##################################################################
final_path = Path(p.path)
prune_indicator = p.prune_indicator
batch_size = p.batch_size
device = p.device
state_dict_path = final_path / 'model.tar'
log_path = final_path / 'hessian.log'
prune_ratio = 0.9  # Just a placeholder


# ##################################################################
# 1. Prepartions
# ##################################################################
# Set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info(state_dict_path)

# Set neccessities
device = torch.device(device)
criterion = nn.CrossEntropyLoss()

# Set the dataset
dataset = CIFAR100Loader()
train_loader, test_loader, _ = dataset.loaders(batch_size=batch_size,
                                               shuffle_train=False,
                                               shuffle_test=False)

# Create the function to set the network
def set_network(prune_indicator,
                prune_ratio,
                state_dict_path,
                device):
    """
    Set a network.
    """
    # Load the dict
    state_dict = torch.load(state_dict_path, map_location=device)

    # Set the network
    net = ResNet(out_dim=100).to(device=device)

    # Prune the network if needed
    if prune_indicator:
        try:
            pruner.global_prune(net, 'l1', prune_ratio, False)
            net = utils.load_state_dict_(net, state_dict)
        except:
            pruner.global_prune(net, 'l1', prune_ratio, True)
            net = utils.load_state_dict_(net, state_dict)
    else:
        net = utils.load_state_dict_(net, state_dict)

    # Load the dict to net
    return net


# ##################################################################
# 1. Calculate the Hessian
# ##################################################################
# Set network
net = set_network(prune_indicator, prune_ratio, state_dict_path, device)

# Set the loader
if p.use_loader == 'train':
    data_loader = train_loader
elif p.use_loader == 'test':
    data_loader = test_loader
else:
    data_loader = train_loader

# Get the data from the loader (we use one random batch)
for data_ in data_loader:
    # Get the data
    inputs, targets, _ = data_

    # Move to the device
    inputs = inputs.to(device=device)
    targets = targets.to(device=device)
    break

# Get logging
logger.info(f'Getting hessian for batch size as {batch_size}...')

# Calculate the trace of hessian
hessian_comp = pyhessian.hessian(net, criterion, data=(inputs, targets), cuda=True)
trace = hessian_comp.trace()

logger.info(f'Trace List: {trace}')
logger.info(f'Trace mean: {np.mean(trace)}\n')
