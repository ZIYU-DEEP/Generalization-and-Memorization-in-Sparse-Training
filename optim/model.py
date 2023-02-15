"""
Title: model.py
Description: The main classes for models, which will load the trainer and save results.
Note: Need to check compatibility.
"""

try:
    import torch_xla.core.xla_model as xm
except:
    pass

from .trainer import Trainer
from .trainer_noisy import NoisyTrainer
from helper import utils, algo
from network.main import build_network
from copy import deepcopy
from shutil import copyfile

import torch
import joblib
import json


# #########################################################################
# 1. Model
# #########################################################################
class Model:
    def __init__(self):
        self.net_name = None
        self.net = None
        self.trainer = None
        self.config = {'optimizer_name': None,
                       'momentum': None,
                       'lr': None,
                       'lr_schedule': None,
                       'lr_milestones': None,
                       'lr_gamma': None,
                       'n_epochs': None,
                       'batch_size': None,
                       'weight_decay': None}

        self.results = {'train_time': None,
                        'test_time': None,
                        'loss_train_list': None,
                        'loss_test_list': None,
                        'acc_train_list': None,
                        'acc_test_list': None,
                        'loss_clean_list': None,
                        'loss_noisy_list': None,
                        'acc_clean_list': None,
                        'acc_generalized_list': None,
                        'acc_memorized_list': None,}

        self.init_method = 'kaiming'

    def set_network(self,
                    net_name='mlp',
                    in_dim: int=12,
                    out_dim: int=2,
                    hidden_act: str='tanh',
                    out_act: str='softmax',
                    hidden_dims: str='10-7-5-4-3',
                    depth: int=32,
                    widen_factor: int=4,
                    dropRate: int=0,
                    growthRate: int=12,
                    compressionRate: int=1):
        """
        Set up the network used for the model.
        """

        self.net_name = net_name
        self.net = build_network(net_name, in_dim, out_dim, hidden_act,
                                 out_act, hidden_dims, depth, widen_factor,
                                 dropRate, growthRate, compressionRate)

        # if device == 'cuda':
        #     self.net = torch.nn.DataParallel(self.net).to(device)
        # self.net = self.net.to(device)

    def init_network(self, init_method: str='kaiming'):
        """
        Initialize the network after it being set up.
        """
        self.init_method = init_method

        if init_method == 'xavier':
            utils.init_weights_xavier(self.net)

        if init_method == 'kaiming':
            utils.init_weights_kaiming(self.net)

    def train(self, dataset, optimizer_name: str='adam', momentum: float=0.9,
              lr: float=0.001, lr_schedule: str='stepwise', lr_milestones: str='50-100',
              lr_gamma: float=0.1, n_epochs: int=500, batch_size: int=256,
              weight_decay: float=1e-6, device: str='cuda', n_jobs_dataloader: int=0,
              fisher_metric: str='fim_monte_carlo', save_fisher:int=1, save_snr:int=1,
              train_noisy: int=0, prune_indicator: int=0, reg_type: str='none',
              reg_weight: float=0.01, final_path: str='./final_path',
              resume_epoch: int=0):
        """
        Call the trainer and get training results.

        Notice that the last parameter prune only helps you to determine the way
        to extract gradients; it is not an indicator if pruning is done during training.
        """

        # Print model information
        print(f'Optimizer: {optimizer_name}; Learning rate: {lr}; Batch size: {batch_size}')

        # ------------- Set up the trainer ------------- #
        if train_noisy:
            self.trainer = NoisyTrainer(optimizer_name, momentum, lr, lr_schedule,
                                        lr_milestones, lr_gamma, n_epochs, batch_size,
                                        weight_decay, device, n_jobs_dataloader,
                                        fisher_metric, save_fisher, save_snr,
                                        prune_indicator, final_path)
        else:
            self.trainer = Trainer(optimizer_name, momentum, lr, lr_schedule,
                                   lr_milestones, lr_gamma, n_epochs, batch_size,
                                   weight_decay, device, n_jobs_dataloader,
                                   fisher_metric, save_fisher, save_snr,
                                   prune_indicator, reg_type, reg_weight,
                                   final_path)

        # ------------- Get the trained network ------------- #
        self.net = self.trainer.train(dataset, self.net, resume_epoch)

        # ------------- Save the results to the result dict ------------- #
        self.results = self.trainer.results  # This is just a lazy mark

        # ------------- Save the config to the config dict ------------- #
        self.config['optimizer_name'] = optimizer_name
        self.config['momentum'] = momentum
        self.config['lr'] = lr
        self.config['lr_schedule'] = lr_schedule
        self.config['lr_milestones'] = lr_milestones
        self.config['lr_gamma'] = lr_gamma
        self.config['n_epochs'] = n_epochs
        self.config['batch_size'] = batch_size
        self.config['weight_decay'] = weight_decay

    def test(self, dataset):
        """
        Test on an arbitrary dataset's test loader.
        """

        self.trainer.test(dataset, self.net)
        self.results['test_time'] = self.trainer.test_time

    def save_net_dict(self, model_path, save_best=1):
        """
        Save the torch model dict.
        """

        if save_best:
            best_model_path = self.trainer.final_path / 'model_best.tar'
            copyfile(best_model_path, model_path)
        else:
            self.net, net_dict = utils.get_net_dict(self.net, self.device)
            torch.save(net_dict, model_path)

    def load_net_dict(self, model_path, device='cuda'):
        """
        Load the torch model dict.
        """

        # Special handling for TPU machine
        if device not in ['cpu', 'cuda']:
            net_dict = torch.load(model_path, map_location='cpu')
        else:
            # Below works for CPU or CUDA
            # Get the net dict from model path
            net_dict = torch.load(model_path, map_location=device)

        # Generically we load a state dict without a module wrapper
        try:
            self.net.load_state_dict(net_dict)
        except:
            try:
                # Handle the case when nn.DataParallel() wrap a module
                # Case 1: '.module' in net but not in dict
                self.net.module.load_state_dict(net_dict)
            except:
                # Case 2: '.module' in dict but not in net
                for key in list(net_dict.keys()):
                    net_dict[key.replace('module.', '')] = net_dict.pop(key)
                self.net.load_state_dict(net_dict)

    def save_results(self, pkl_path):
        """
        Save the training results.
        """

        joblib.dump(self.results, pkl_path)

    def load_results(self, pkl_path):
        """
        Load the training results.
        """

        self.results = joblib.load(pkl_path)

    def save_config(self, json_path):
        """
        Save the configurations for training.
        """

        with open(json_path, 'w') as fp:
            json.dump(self.config, fp)
