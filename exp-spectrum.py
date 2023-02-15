"""
[Title] experiment-spectrum.py
[Usage] This is a file to calculate the hessian spectrum.
"""

from helper import utils, pruner, hessian
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
                    help='The ratio for sparse training.')
parser.add_argument('-no', '--no', type=int, default=0,
                    help='The epoch number to be loaded from the state_dicts folder.')

# Better to leave them as default
parser.add_argument('-ul', '--use_loader', type=str, default='test',
                    help='The loader to use in evaluating the fisher.',
                    choices=['train', 'test', 'clean', 'noisy'])
parser.add_argument('-dv', '--device', type=str, default='cuda',
                    help='Choose from cpu, cuda, and tpu.')
parser.add_argument('-bs', '--batch_size', type=int, default=128,
                    help='The batch size for training.')
parser.add_argument('-ipd', '--init_poly_deg', type=int, default=64,
                    help='The iterations used to compute spectrum range.')
parser.add_argument('-pd', '--poly_deg', type=int, default=256,
                    help='The higher the parameter the better the approximation.')

p = parser.parse_args()


# ##################################################################
# 0. Define Global Variables
# ##################################################################
final_path = Path(p.path)
prune_indicator = p.prune_indicator
batch_size = p.batch_size
device = p.device
state_dict_path = final_path / 'state_dicts' / f'epoch_{p.no}.pkl'
eigen_path = final_path / 'eigen_results'
log_path = final_path / 'hessian_spectrum.log'
prune_ratio = p.prune_ratio  # Just a placeholder

if not os.path.exists(eigen_path): os.makedirs(eigen_path)

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

# Get logging
logger.info(f'Getting hessian for batch size as {batch_size}...')

# Wrap the Hessian class
H = hessian.Hessian(loader=data_loader,
                    model=net,
                    hessian_type='Hessian')

# Get the Hessian eigenvalue and the associated density
H_eigval, H_eigval_density = H.LanczosApproxSpec(init_poly_deg=p.init_poly_deg,
                                                 poly_deg=p.poly_deg)

# Save the eigenvalue and the corresponding density
np.save(eigen_path / f'hessian_eigval_{p.no}', H_eigval)
np.save(eigen_path / f'hessian_eigval_density_{p.no}', H_eigval_density)
logger.info(f'The spectrum is now saved in npz files.')


# ##################################################################
# 2. Plot the spectrum
# ##################################################################
# Plot the hessian spectrum
plt.figure(figsize=(15, 5.3))
plt.semilogy(H_eigval, H_eigval_density,  color='darkorange', lw=3, alpha=0.8)
plt.ylabel('Density of Hessian Spectrum', fontsize=20)
plt.xlabel('Eigenvalue', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
sea.despine()
plt.savefig(eigen_path / f'hessian_spectrum_{p.no}.pdf')

# Log the results
logger.info('Done!')
