"""
Title: trainer_noisy.py
Description: A simple trainer for noisy setting.
"""

from helper import utils, algo
from .base_trainer import BaseTrainer
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ExponentialLR
from nngeometry.object import PMatKFAC, PMatEKFAC, PMatDiag, PVector
from collections import OrderedDict
from copy import deepcopy

import torch.optim as optim
import torch.nn as nn
import torch.nn.parallel
import logging
import torch
import time
import sys

# ----------- Handle the TPU Training Framework ----------- #
try:
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    def len_parallelloader(self):
        return len(self._loader._loader.sampler)
    pl.PerDeviceLoader.__len__ = len_parallelloader
    torch.set_default_tensor_type('torch.FloatTensor')
except:
    pass
# --------------------------------------------------------- #


# #########################################################################
# 1. Trainer
# #########################################################################
# #########################################################################
# 1. Trainer
# #########################################################################
class NoisyTrainer(BaseTrainer):
    def __init__(self,
                 optimizer_name: str='sgd',
                 momentum: float=0.9,
                 lr: float=0.01,
                 lr_schedule: str='exponential',
                 lr_milestones: str='50-100',
                 lr_gamma: float=0.1,
                 n_epochs: int=60,
                 batch_size: int=12,
                 weight_decay: float = 1e-6,
                 device: str='cuda',
                 n_jobs_dataloader: int=0,
                 fisher_metric: str='fim_monte_carlo',
                 save_fisher: int=1,
                 save_snr: int=1,
                 prune_indicator: int=0,
                 final_path: str='./final_path'):
        """
        A trainer for model. You should only feed in noisy data loader here.

        Notice that the last parameter prune only helps you to determine the way
        to extract gradients; it is not an indicator if pruning is done during training.
        """

        super().__init__(optimizer_name, momentum, lr, lr_schedule, lr_milestones, lr_gamma,
                         n_epochs, batch_size, weight_decay, device, n_jobs_dataloader,
                         fisher_metric, save_fisher, save_snr, prune_indicator, final_path)

        # Initialize statistics for training results
        self.best_test_acc = 0
        self.results = {'train_time': None,
                        'test_time': None,
                        'loss_train_list': [],
                        'loss_test_list': [],
                        'acc_train_list': [],
                        'acc_test_list': [],
                        'loss_clean_list': [],
                        'loss_noisy_list': [],
                        'acc_clean_list': [],
                        'acc_generalized_list': [],
                        'acc_memorized_list': []}

    def train(self, dataset, net, resume_epoch):
        # Get the logger
        logger = logging.getLogger()
        logger.info(self.device)

        # Set the dataset
        # ==================== Configure device for dataset =================== #
        train_loader_base, test_loader_base, _ = dataset.loaders(batch_size=self.batch_size,
                                                       num_workers=self.n_jobs_dataloader)

        # ==================== Configure device for network =================== #
        # Set the device for network
        if self.device == 'cuda':
            net = torch.nn.DataParallel(net).to(self.device)
        elif self.device == 'cpu':
            net = net.to(self.device)
        else:  # already outside as xm.xla_device() in run.py
            logger.info('Set network to tpu!')
            net = xmp.MpModelWrapper(net)
            net = net.to(device=self.device)
            self.lr *= xm.xrt_world_size()
        logger.info(f'Device: {self.device}')

        # Use cross-entropy as the default classification criterion
        criterion = nn.CrossEntropyLoss()

        # Set the optimizer
        logger.info(f'Optimizer: {self.optimizer_name}!')
        if self.optimizer_name == 'adam':
            optimizer = optim.Adam(net.parameters(),
                                   lr=self.lr,
                                   betas=(0.9, 0.999),
                                   weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(net.parameters(),
                                  lr=self.lr,
                                  momentum=self.momentum,
                                  weight_decay=self.weight_decay)

        # Set up timer
        start_time = time.time()

        # Set the learning rate scheduler
        if self.lr_schedule == 'stepwise':
            scheduler = MultiStepLR(optimizer,
                                    milestones=self.lr_milestones,
                                    gamma=self.lr_gamma)
        elif self.lr_schedule == 'exponential':
            scheduler = ExponentialLR(optimizer,
                                      gamma=self.lr_gamma)

        # Handle when training is resumed
        if resume_epoch:
            # Handle the lr scheuler
            for i in range(resume_epoch + 1):
                scheduler.step()
            logger.info('LR scheduler: resumed learning rate is %g' %
                         float(scheduler.get_last_lr()[0]))

            # Fill in the results
            resume_path = self.final_path / 'state_dicts' / f'epoch_{resume_epoch}.pkl'
            re_results = joblib.load(self.final_path / 'results.pkl')
            ind = resume_epoch + 1
            self.results['loss_train_list'] = re_results['loss_train_list'][:ind]
            self.results['loss_test_list'] = re_results['loss_test_list'][:ind]
            self.results['acc_train_list'] = re_results['acc_train_list'][:ind]
            self.results['acc_test_list'] = re_results['acc_test_list'][:ind]
            self.best_test_acc = max(self.results['acc_test_list'])

        # Create an empty dictionary to save
        grad_dict = {f'epoch {i}': {'weight_norm': [],
                                    'grad_mean': [],
                                    'grad_std': []}
                      for i in range(self.n_epochs)}
        fisher_dict = OrderedDict()

        # Start training
        logger.info('Starting training...')
        for epoch in range(self.n_epochs):

            # ========== Handle when training is resumed ============= #
            if resume_epoch:
                # Accumulate the epoch number
                epoch += resume_epoch + 1
                if epoch >= self.n_epochs:
                    break

            # ========== Track lr and time ============= #
            # Record the time when learning rate changes
            if epoch in self.lr_milestones:
                logger.info('LR scheduler: new learning rate is %g' %
                            float(scheduler.get_last_lr()[0]))

            # Record time
            epoch_start_time = time.time()

            # ==================== Save state dicts =================== #
            start_time_save = time.time()  # debug time
            net_dict = utils.get_net_dict(net, self.device)

            # Save net dicts; the function makes you save less in later epochs
            utils.save_net_dict(net_dict=net_dict,
                                epoch=epoch,
                                is_last_epoch=epoch == self.n_epochs - 1,
                                path=self.final_path / 'state_dicts' / f'epoch_{epoch}.pkl')
            end_time_save = time.time()  # debug time

            # ======================= Training  ======================= #
            # Split data into batches; update weight for each batch
            start_time_train = time.time()  # debug time

            # Training the network
            net.train()

            # You have to initiate a new loader each time for tpu
            if self.device not in ['cuda', 'cpu']:
                # Make train loader parallel for TPU
                train_loader = pl.ParallelLoader(train_loader_base, [self.device])
                train_loader = train_loader.per_device_loader(self.device)
            else:
                train_loader, test_loader = train_loader_base, test_loader_base

            for data in train_loader:
                # Set up data; compared to regular trainer, we add two more _
                inputs, _, _, y_noisy, _ = data

                # Move data to device; TPU has taken care of it at start
                if self.device in ['cpu', 'cuda']:
                    inputs, y_noisy = inputs.to(self.device), y_noisy.to(self.device)

                # Get prediction results and losses
                outputs = net(inputs)
                losses_train = criterion(outputs, y_noisy)

                # Backpropagation
                optimizer.zero_grad()
                losses_train.backward()

                # Step optimizer
                if self.device in ['cpu', 'cuda']:
                    optimizer.step()
                else:
                    xm.optimizer_step(optimizer)

                # Delete unnecessary vals
                del data, inputs, outputs, losses_train, y_noisy

            end_time_train = time.time()  # debug time

            # ================= Get training and testing stats ============= #
            # Set up the general statistics
            loss_train, loss_test = utils.Tracker(), utils.Tracker()
            acc_train, acc_test = utils.Tracker(), utils.Tracker()

            # [Only for train loader] Set up statistics for noisy setting
            loss_clean, loss_noisy = utils.Tracker(), utils.Tracker()
            acc_clean = utils.Tracker()
            acc_generalized = utils.Tracker()  # correct += 1 if pred y is true y
            acc_memorized = utils.Tracker()  # correct += 1 if pred y is noisy y

            # debug time
            start_time_test = time.time()
            net.eval()

            # You have to initiate a new loader each time for TPU
            if self.device not in ['cuda', 'cpu']:
                # Make train loader parallel for TPU
                train_loader = pl.ParallelLoader(train_loader_base, [self.device])
                train_loader = train_loader.per_device_loader(self.device)
                # Make test loader parallel for TPU
                test_loader = pl.ParallelLoader(test_loader_base, [self.device])
                test_loader = test_loader.per_device_loader(self.device)

            with torch.no_grad():
                # -------------- Get Train Accuracy and losses ---------- #
                # Start getting the training results
                for data in train_loader:
                    # Set up data
                    inputs, y, _, y_noisy, is_noisy = data
                    if self.device in ['cpu', 'cuda']:
                        inputs, y = inputs.to(self.device), y.to(self.device)
                        y_noisy = y_noisy.to(self.device)
                        is_noisy = is_noisy.to(self.device)

                    # Compute prediction error
                    outputs = net(inputs)
                    losses = criterion(outputs, y_noisy)

                    # Get batch-wise statistics
                    size_ = len(y)  # The number of samples in this batch
                    loss_train.update(losses, size_)
                    acc_train.update(utils.get_acc(outputs, y_noisy, size_), size_)

                    # Get statistics for the noisy setting
                    # Get accuracy in clean data
                    # Predictions for all clean data
                    outputs_clean = outputs[~is_noisy.bool()]
                    # True labels for all clean data
                    y_true_clean = y[~is_noisy.bool()]
                    # Size of this clean portion
                    size_clean = len(y_true_clean)
                    # Calculate accuracy
                    acc_clean.update(utils.get_acc(outputs_clean, y_true_clean,
                                                   size_clean), size_clean)

                    # Get accuracy in noisy data
                    # Predictions for all perturbed data
                    outputs_noisy = outputs[is_noisy.bool()]
                    # True labels for all perturbed data
                    y_true_noisy = y[is_noisy.bool()]
                    # Fake labels for all perturbed data
                    y_fake_noisy = y_noisy[is_noisy.bool()]
                    # Size of this noisy portion
                    size_noisy = len(y_true_noisy)
                    # Calculate the portion of generalized
                    acc_generalized.update(utils.get_acc(outputs_noisy, y_true_noisy,
                                           size_noisy), size_noisy)
                    # Calculate the portion of memorized
                    acc_memorized.update(utils.get_acc(outputs_noisy, y_fake_noisy,
                                         size_noisy), size_noisy)

                    # Calculate loss on clean and noisy portion respectively
                    losses_clean = criterion(outputs_clean, y_true_clean)
                    losses_noisy = criterion(outputs_noisy, y_fake_noisy)
                    loss_clean.update(losses_clean, size_clean)
                    loss_noisy.update(losses_noisy, size_noisy)

                    del inputs, outputs, y, losses, y_noisy, is_noisy

                # -------------- Get Test Accuracy and losses ----------- #
                for data_ in test_loader:
                    # Set up data
                    inputs_, y_, _, _, _ = data_
                    if self.device in ['cpu', 'cuda']:
                        inputs_, y_ = inputs_.to(self.device), y_.to(self.device)

                    # Compute prediction error
                    outputs_ = net(inputs_)
                    losses_ = criterion(outputs_, y_)

                    # Get batch-wise statistics
                    size_ = len(y_)  # The number of samples in this batch
                    loss_test.update(losses_, size_)
                    acc_test.update(utils.get_acc(outputs_, y_, size_), size_)
                    del inputs_, outputs_, losses_

            end_time_test = time.time()  # debug time

            # Get time
            epoch_train_time = time.time() - epoch_start_time

            # Step in scheduler
            scheduler.step()

            # ======================= Get SNR  ======================= #
            if self.device not in ['cuda', 'cpu']:
                # Make train loader parallel for TPU
                train_loader = pl.ParallelLoader(train_loader_base, [self.device])
                train_loader = train_loader.per_device_loader(self.device)

            start_time_grad = time.time()  # debug time
            if self.save_snr:
                logger.info('Not implemented yet; update the utils.get_snr first!')
                # utils.get_snr(net, train_loader, criterion, optimizer, grad_dict,
                #               epoch, self.device, self.final_path, self.prune_indicator)
            end_time_grad = time.time()  # debug time

            # ================ Get Fisher Information  =============== #
            start_time_fisher = time.time()
            if self.device not in ['cuda', 'cpu']:
                # Make train loader parallel for TPU
                train_loader = pl.ParallelLoader(train_loader_base, [self.device])
                train_loader = train_loader.per_device_loader(self.device)

            if self.save_fisher:
                fisher_dict[epoch] = utils.get_fisher(net=net,
                                                      data_loader=train_loader,
                                                      representation=PMatEKFAC,
                                                      fisher_metric=self.fisher_metric,
                                                      n_output=y_.shape[-1],
                                                      variant='classif_logits',
                                                      device=self.device,
                                                      final_path=self.final_path)
                logger.info(f'fisher trace: {fisher_dict[epoch]}')  # Debug
            end_time_fisher = time.time()

            # ================ Record performance in list  =============== #
            # Get epoch-wise statistics
            start_time_append = time.time() # ddebug
            self.results['loss_train_list'].append(loss_train.mean)
            self.results['loss_test_list'].append(loss_test.mean)
            self.results['acc_train_list'].append(acc_train.mean)
            self.results['acc_test_list'].append(acc_test.mean)

            # Get the performance results for noisy training data
            start_time_append = time.time() # ddebug
            self.results['loss_clean_list'].append(loss_clean.mean)
            self.results['loss_noisy_list'].append(loss_noisy.mean)
            self.results['acc_clean_list'].append(acc_clean.mean)
            self.results['acc_generalized_list'].append(acc_generalized.mean)
            self.results['acc_memorized_list'].append(acc_memorized.mean)


            # ============== Save essentials for plots  ============== #
            # Save in log
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} '
                        f'| Train Loss: {self.results["loss_train_list"][-1]:.6f} '
                        f'| Train Acc: {self.results["acc_train_list"][-1]:.3f} '
                        f'| Test Loss: {self.results["loss_test_list"][-1]:.6f} '
                        f'| Test Acc: {self.results["acc_test_list"][-1]:.3f} '
                        f'| Time: {time.time() - epoch_start_time:.3f}s '
                        f'| Memory: {utils.get_memory():.3f} GB |')

            logger.info(f'| Loss clean: {self.results["loss_clean_list"][-1]:.3f} '
                        f'| Loss noisy: {self.results["loss_noisy_list"][-1]:.3f} '
                        f'| Acc clean: {self.results["acc_clean_list"][-1]:.3f} '
                        f'| Acc generalized: {self.results["acc_generalized_list"][-1]:.3f} '
                        f'| Acc memorized: {self.results["acc_memorized_list"][-1]:.3f} ')

            # logger.info(f'| Time | Train: {end_time_train - start_time_train:.3f}s '
            #             f'| Grad: {end_time_grad - start_time_grad:.3f}s '
            #             f'| Fisher: {end_time_fisher - start_time_fisher:.3f}s '
            #             f'| Test: {end_time_test - start_time_test:.3f}s '
            #             f'| Save: {end_time_save - start_time_save:.3f}s ')

            # ============== Save the best model ============== #
            if self.results['acc_test_list'][- 1] > self.best_test_acc:
                self.best_test_acc = self.results['acc_test_list'][-1]
                net_dict = utils.get_net_dict(net, self.device)
                torch.save(net_dict, self.final_path / 'model_best.tar')

            # Save the results
            joblib.dump(self.results, self.final_path / 'results.pkl')

        # ============== Save post-training information  ============== #
        # Save statistics
        self.results['train_time'] = time.time() - start_time

        if self.save_fisher:
            torch.save(fisher_dict, self.final_path / 'fisher_dict.pkl')
        if self.save_snr:
            torch.save(grad_dict, self.final_path / 'grad_dict.pkl')

        # Logging final information
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')
        return net

    def test(self, dataset, net):
        # Get the logger
        logger = logging.getLogger()

        # Log results
        logger.info('The lazy person did not implement this. : )')
