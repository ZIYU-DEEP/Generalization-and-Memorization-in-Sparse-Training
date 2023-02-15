"""
[Title] run.py
[Usage] The file to train a model and save model information.
[Functions to be added]
    [ ] iterative training
    [ ] sparsity-aware initialization
    [ ] random pruning (random mask?)
    [ ] SNIP pruning
"""

from loader import load_dataset
from optim import Model
from helper import pruner, plotter, utils
from pathlib import Path

import os
import time
import torch
import random
import logging
import argparse
import numpy as np

# ----------- Handle the TPU Training Framework ----------- #
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except:
    pass
# --------------------------------------------------------- #


# ############################################
# 1. Preparation
# ############################################
# ===========================================
# 1.1. Parameters
# ===========================================
# Initialize the parser
parser = argparse.ArgumentParser()

# Arguments for general training mechanism
# Sparse finetuning is dense-prune-finetune, and sparse training is prune-finetune.
parser.add_argument('-m', '--mode', type=str, default='dense',
                    help='Set the training mode',
                    choices=['dense', 'sparse-finetune',
                             'sparse-scratch', 'lottery'])
parser.add_argument('-pm', '--prune_method', type=str, default='l1',
                    help='Set the training mode',
                    choices=['l1'])
parser.add_argument('-pr', '--prune_ratio', type=float, default=0.9,
                    help='The ratio for sparse training')
parser.add_argument('-pl', '--prune_last', type=int, default=0,
                    help='0 if not pruning for the last layer, else 1.')
parser.add_argument('-rr', '--results_root', type=str, default='./results',
                    help='The directory to save results.')

# Arguments for loading the pre-trained model
parser.add_argument('-lp', '--load_pretrain', type=int, default=1,
                    help='To finetune, 1 if load a trained dense, else train a new dense')
parser.add_argument('-pp', '--pretrain_model_path', type=str,
                    default='./results/loader/dense_folder/model.tar',
                    help='The path for a pretrained dense model.')
parser.add_argument('-tp', '--lottery_model_path', type=str,
                    default='./results/loader/dense_folder/state_dicts/epoch_0.pkl',
                    help='The path for a pretrained dense model.')

# Arguments for loading datasets
parser.add_argument('-ln', '--loader_name', type=str, default='toy',
                    help='The name for your dataset.',
                    choices=['toy', 'mnist', 'cifar10', 'cifar100',
                             'fmnist', 'cifar100_tpu',
                             'cifar10_noisy', 'fmnist_noisy', 'tiny_imagenet'])
parser.add_argument('-rt', '--root', type=str, default='./data/',
                    help='The root path for stored data.')
parser.add_argument('-fn', '--filename', type=str, default='toy',
                    help='The filename for your data, e.g., toy, MNIST or CIFAR10.')
parser.add_argument('-ts', '--test_size', type=float, default=0.2,
                    help='The test split ratio for the toy dataset.')
parser.add_argument('-rs', '--random_state', type=int, default=42,
                    help='Use 0 to let the machine set a random random seed for you.')
parser.add_argument('-dl', '--download', type=int, default=0,
                    help='1 if you need download the dataset, otherwise 0.')

# Arguments for setting network
parser.add_argument('-nt', '--net_name', type=str, default='mlp',
                    help='The name for your network',
                    choices=['mlp', 'alexnet', 'preresnet', 'resnet',
                             'densenet', 'vgg11', 'vgg11_bn', 'vgg13',
                             'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19',
                             'vgg19_bn', 'mnist_lenet', 'mnist_alexnet'])
parser.add_argument('-in', '--in_dim', type=int, default=12,
                    help='The feature dimension for the input data X.')
parser.add_argument('-ot', '--out_dim', type=int, default=2,
                    help='The number of classes of the output data y.')
parser.add_argument('-ha', '--hidden_act', type=str, default='tanh',
                    help='The activation for hidden layers, e.g., tanh, relu, softmax, sigmoid.')
parser.add_argument('-oa', '--out_act', type=str, default='softmax',
                    help='The activation for the output layer, e.g., softmax, sigmoid.')
parser.add_argument('-hd', '--hidden_dims', type=str, default='10-7-5-4-3',
                    help='The hidden dimensions for MLP; using hypen to connect numbers.')
parser.add_argument('-dp', '--depth', type=int, default=32,
                    help='The depth for preresnet or densenet.')
parser.add_argument('-wf', '--widen_factor', type=int, default=4,
                    help='The widen factor for resnet.')
parser.add_argument('-dr', '--dropRate', type=int, default=0,
                    help='The drop rate for dense net.')
parser.add_argument('-gr', '--growthRate', type=int, default=12,
                    help='The growth rate for dense net.')
parser.add_argument('-cr', '--compressionRate', type=int, default=1,
                    help='The compression rate for densenet.')

# Arguments for the model
parser.add_argument('-im', '--init_method', type=str, default='kaiming',
                    help='Currently support kaiming, xavier, and none.')
parser.add_argument('-on', '--optimizer_name', type=str, default='adam',
                    help='Currently support sgd or adam.')
parser.add_argument('-mm', '--momentum', type=float, default=0.9,
                    help='The momentum of the optimizer.')
parser.add_argument('-lr', '--lr', type=float, default=0.0004,
                    help='The learning rate for optimization.')
parser.add_argument('-ls', '--lr_schedule', type=str, default='stepwise',
                    help='Currently support stepwise or exponential.')
parser.add_argument('-lm', '--lr_milestones', type=str, default='50-100',
                    help='The milestones for stepwise lr scheduler.')
parser.add_argument('-lg', '--lr_gamma', type=float, default=0.1,
                    help='The decay rate for learning rate.')
parser.add_argument('-ne', '--n_epochs', type=int, default=500,
                    help='The number of training epochs.')
parser.add_argument('-bs', '--batch_size', type=int, default=256,
                    help='The batch size for training.')
parser.add_argument('-wt', '--weight_decay', type=float, default=1e-6,
                    help='The decay rate for weight.')
parser.add_argument('-dv', '--device', type=str, default='cpu',
                    choices=['cpu', 'cuda', 'tpu'])
parser.add_argument('-nj', '--n_jobs_dataloader', type=int, default=0,
                    help='The number of workers for jobloader; default is just 0.')

# Arguments for the loss regularizer
parser.add_argument('-rp', '--reg_type', type=str, default='none',
                    choices=['none', 'jacobian'],
                    help='The type of loss regularization.')
parser.add_argument('-rw', '--reg_weight', type=float, default='0.01',
                    help='The coefficient for the loss regularizer.')

# Arguments for the subset selection algorithm
parser.add_argument('-sub', '--train_subset', type=int, default=0,
                    help='0 if use subset selection algorithm in training.')
parser.add_argument('-subr', '--subset_ratio', type=float, default=0.1,
                    help='The ratio to sample from the pool.')
parser.add_argument('-subm', '--subset_method', type=str, default='loss',
                    help='The method for subset selection.',
                    choices=['default', 'uniform', 'loss', 'fisher',
                             'reducible_loss', 'reducible_fisher'])

# Arguments for calculating fisher metric
parser.add_argument('-fd', '--fisher_method', type=str, default='ours',
                    help='Ours is written by us; nngeometry is to use existing package.',
                    choices=['vmap', 'ours', 'nngeometry'])  # nngeometry may not work on the pruned
parser.add_argument('-fm', '--fisher_metric', type=str, default='fim_monte_carlo',
                    help='fim if closed-form results, or fim_monte_carlo with samples.')
parser.add_argument('-ms', '--monte_carlo_samples', type=int, default=1000,
                    help='The number of monte carlo trials for sampling X.')
parser.add_argument('-mt', '--monte_carlo_trials', type=int, default=5,
                    help='The number of monte carlo trials for sampling predicted y.')
parser.add_argument('-bf', '--batch_size_fisher', type=int, default=1,
                    help='Must be set to be 1 for ours, see utils.get_fisher_ours().')

# Arguments for the indicators of saving things or training config
parser.add_argument('-sb', '--save_best', type=int, default=1,
                    help='1 if save the best model, else 0.')
parser.add_argument('-sf', '--save_fisher', type=int, default=1,
                    help='1 if save fisher, else 0.')
parser.add_argument('-ss', '--save_snr', type=int, default=1,
                    help='1 if save snr, else 0.')

# Arguments for noisy setting
parser.add_argument('-tn', '--train_noisy', type=int, default=0,
                    help='1 if training in noisy setting, else 0.')
parser.add_argument('-nm', '--noise_method', type=str, default='symmetric',
                    help='Check the noisy loader for details.')
parser.add_argument('-nr', '--noise_ratio', type=float, default=0.2,
                    help='The ratio of flipped labels in the data set.')

# Arguments for loading pretrain (a dense model) path
parser.add_argument('-lrp', '--lr_pretrain', type=float, default=0.0004)
parser.add_argument('-lsp', '--lr_schedule_pretrain', type=str, default='stepwise')
parser.add_argument('-lmp', '--lr_milestones_pretrain', type=str, default='50-100')
parser.add_argument('-lgp', '--lr_gamma_pretrain', type=float, default=0.1)
parser.add_argument('-nep', '--n_epochs_pretrain', type=int, default=0)
parser.add_argument('-bsp', '--batch_size_pretrain', type=int, default=256)

# Arguments for resuming training
# Some clusters have a 4-hour GPU usage limit
# If your training is interrupted, you may just resume it
# by telling the code the epoch to resume
# Notice no other parameter needed to change
parser.add_argument('-ree', '--resume_epoch', type=int, default=0,
                    help='The epoch to resume the training')

p = parser.parse_args()

# ===========================================
# 1.2. Setup random states and devices
# ===========================================
# Set a random random seed
if not p.random_state:
    random_state = np.random.randint(low=0,
                                     high=np.iinfo(np.int32).max,
                                     size=1)[0]
# Use the input argument as the random state
else:
    random_state = p.random_state

random.seed(random_state)
torch.manual_seed(random_state)
np.random.seed(random_state)

# Set up device
if p.device == 'tpu':
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
else:
    device = p.device  # Just keep it as string for convenience

# ===========================================
# 1.3. Define Path
# ===========================================
# Get the time
time_now = time.ctime().split()[1:4]

# Configure the network string for path
net_str = f'{p.net_name}'
if p.net_name == 'mlp': net_str += f'-{p.hidden_act}-{p.hidden_dims}'
if p.net_name == 'resnet': net_str += f'-{p.depth}-{p.widen_factor}'
if p.net_name == 'densenet': net_str += f'-{p.dropRate}-{p.growthRate}-{p.compressionRate}'

# Configure the name for noisy training
noise_str = ''
if p.train_noisy:
    noise_str += f'noise_{p.noise_method}_{p.noise_ratio}_'

# Configure the pruning specifics for the folder name
prune_str = ''
if p.mode != 'dense':
    prune_str += f'method_{p.prune_method}_ratio_{p.prune_ratio}_'

# Configure other optimization specifics for the folder name
epoch_str = f'epochs_{p.n_epochs}_'
batch_str = f'batch_{p.batch_size}_'
optim_str = f'{p.optimizer_name}_'
optim_str += f'lr_{p.lr}-{p.lr_schedule}-{p.lr_milestones}-{p.lr_gamma}_'
optim_str += f'mm_{p.momentum}_w_decay_{p.weight_decay}_init_{p.init_method}_'
seed_str = f'seed_{random_state}'
time_str = f'time_{time_now[0]}-{time_now[1]}-{time_now[2]}'

# Set the folder name
folder_name = f'{noise_str}{prune_str}{epoch_str}{batch_str}{optim_str}{seed_str}'

# Append the folder name with the regularzier type
if p.reg_type != 'none':
    folder_name += f'_reg_{p.reg_type}-{p.reg_weight}'

# Append the folder name with the subset selection algorithm type
if p.train_subset:
    folder_name += f'_train_subset-{p.subset_ratio}-{p.subset_method}'

# Set the folder name for sparse training with dense pretraining
if p.mode != 'dense':
    folder_name += f'_epoch_pre_{p.n_epochs_pretrain}'
    folder_name += f'_batch_pre_{p.batch_size_pretrain}'
    folder_name += f'_lr_pre_{p.lr_pretrain}-{p.lr_schedule_pretrain}'
    folder_name += f'-{p.lr_milestones_pretrain}-{p.lr_gamma_pretrain}'
    prune_indicator = 1
else:
    prune_indicator = 0

# Uncomment the following if you want the folder name with time string
# folder_name += f'_{time_str}'

# Set the final path
final_path = Path(p.results_root) / f'{p.loader_name}'\
             / net_str / f'{p.mode}' / folder_name

# Set the individual path for files inside final path
log_path = final_path / 'training.log'
model_path = final_path / 'model.tar'
results_path = final_path / 'results.pkl'
config_path = final_path / 'config.json'
state_dicts_path = final_path / 'state_dicts'
resume_path = state_dicts_path / f'epoch_{p.resume_epoch}.pkl'
fisher_plot_path = final_path / f'fisher_{p.mode}.pdf'
grad_plot_path = final_path / f'gradients_{p.mode}.pdf'
weight_plot_path = final_path / f'weights_{p.mode}.pdf'
performance_plot_path = final_path / f'performance_{p.mode}.pdf'
performance_noisy_plot_path = final_path / f'performance_noisy_{p.mode}.pdf'
grad_plot_smoothed_path = final_path / f'gradients_smoothed_{p.mode}.pdf'
grad_plot_caption = str(final_path)

if not os.path.exists(final_path): os.makedirs(final_path)
if not os.path.exists(state_dicts_path): os.makedirs(state_dicts_path)

# ===========================================
# 1.4. Setup Logger
# ===========================================
logger = utils.set_logger(log_path)
logger.info(f'Random state: {random_state}')
logger.info(final_path)
logger.info(time_str)


# ############################################
# 2. Model Training
# ############################################
# ---------------------- Configure dataset ---------------------- #
dataset = load_dataset(p.loader_name,
                       p.root,
                       p.filename,
                       p.test_size,
                       p.noise_method,
                       p.noise_ratio,
                       random_state,
                       p.download)

# ------------------ Set up model with trainer ------------------ #
model = Model()
model.set_network(p.net_name, p.in_dim, p.out_dim, p.hidden_act,
                  p.out_act, p.hidden_dims, p.depth, p.widen_factor,
                  p.dropRate, p.growthRate, p.compressionRate)

# ---------------------- Set training mode (resuming) ---------------------- #
if p.resume_epoch:
    logger.info(f'Resuming training from {resume_path}.')
    if p.mode == 'dense':
        # Directly load the previous results and net dict
        model.load_net_dict(resume_path, device)

    if p.mode in ['sparse-scratch', 'sparse-finetune', 'lottery']:
        # Just to make sure the net dict keys are matched
        pruner.global_prune(model.net, p.prune_method, p.prune_ratio, p.prune_last)

        # Load the previous results and net dict
        model.load_net_dict(resume_path, device)

# ---------------------- Set training mode (non-resuming) ---------------------- #
if not p.resume_epoch:
    # Option 1: Dense training
    if p.mode == 'dense':
        # Initialize the network
        model.init_network(p.init_method)

    # Option 2: Sparse finetuning / sparse training from scratch
    if p.mode in ['sparse-scratch', 'sparse-finetune', 'lottery']:
        assert p.load_pretrain, 'for now, please pre-train a dense net to get mask.'

        # ---------------------- A. Sparse Finetuning ---------------------- #
        # Load the pre-trained model path
        logger.info(f'Loading from {p.pretrain_model_path}.')
        model.load_net_dict(p.pretrain_model_path, device)

        # Prune the pretrained network
        pruner.global_prune(model.net, p.prune_method, p.prune_ratio, p.prune_last)

        # ----------------- B. Sparse Training from Scratch ---------------- #
        # Re-intialize the weights if you choose sparse training from scratch
        if p.mode == 'sparse-scratch':
            print('Initialize for sparse training from scratch.')
            model.init_network(p.init_method)

        # ------------------------ C. Lottery Tickets ---------------------- #
        # Re-use the dense weight at initialization for lottery tickets
        if p.mode == 'lottery':
            # Create a new model instance for lottery
            lottery_model = Model()
            lottery_model.set_network(p.net_name, p.in_dim, p.out_dim,
                                      p.hidden_act, p.out_act, p.hidden_dims,
                                      p.depth, p.widen_factor, p.dropRate,
                                      p.growthRate, p.compressionRate)

            # Use the initialization of the dense model
            logger.info(f'Loading from {p.lottery_model_path}.')
            lottery_model.load_net_dict(p.lottery_model_path, device)

            # Prune the lottery model with the mask from the pretrained model
            pruner.global_prune_custom_mask(lottery_model.net, model.net)

            # Cover the name for the model
            model = lottery_model

# ---------------------- Training the model --------------------- #
model.train(dataset, p.optimizer_name, p.momentum, p.lr, p.lr_schedule,
            p.lr_milestones, p.lr_gamma, p.n_epochs, p.batch_size,
            p.weight_decay, device, p.n_jobs_dataloader,
            p.fisher_metric, p.save_fisher, p.save_snr,
            p.train_noisy, prune_indicator, p.reg_type, p.reg_weight,
            final_path, p.train_subset, p.subset_ratio, p.subset_method,
            p.resume_epoch)


# ############################################
# 3. Save Statistics
# ############################################
# Save net dicts, results and configs
model.save_net_dict(model_path, p.save_best)
model.save_config(config_path)

# Handle the net dicts saved by TPU
if p.device == 'tpu':
    utils.save_net_dicts_tpu(model.net, final_path, p.n_epochs, device)

# ############################################
# 4. Plot for gradients
# ############################################
# Annouce the plotting stage
logger.info('Stay tuned: your little Van Gogh is drawing...')

# Save the performance plot
plotter.plot_performance(final_path, performance_plot_path, grad_plot_caption)

if p.save_snr:
    plotter.plot_gradients(final_path, grad_plot_path, grad_plot_caption, p.save_fisher, 0.0)
    plotter.plot_gradients(final_path, grad_plot_smoothed_path, grad_plot_caption, p.save_fisher, 0.6)  # Smoothed lines
    plotter.plot_weights(final_path, weight_plot_path, grad_plot_caption, 0.0)

# Plot fisher information if saved
if p.save_fisher:
    plotter.plot_fisher(final_path, fisher_dict_path, fisher_plot_path)

if p.train_noisy:
    plotter.plot_performance_noisy(final_path, performance_noisy_plot_path)

logger.info('All done. Good luck!')
logger.info(str(final_path))
