"""
[Title] utils.py
[Use] A general helper file.
[TOC] 1. General helper functions;
      2. Helpers for networks;
      3. Helpers for optimizers;
      4. Calculating SNR;
      5. Calculating Fisher information.
"""

from network.main import build_network
from .pruner import global_prune

from nngeometry.layercollection import LayerCollection
from nngeometry.metrics import FIM, FIM_MonteCarlo
from collections import OrderedDict
import torch.nn.functional as F
from functools import reduce
from torch import nn

import logging
import psutil
import torch
import time
import os

# ----------- Handle the TPU Training Framework ----------- #
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except:
    pass

try:
    from functorch.experimental import replace_all_batch_norm_modules_
    from functorch import make_functional_with_buffers, vmap, grad
except:
    pass
# --------------------------------------------------------- #


# #########################################################################
# 1. General Helper Functions
# #########################################################################
def set_logger(log_path):
    """
    Set up a logger for use.
    """
    # Config the logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    # Set level and formats
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Record logging
    logger.info(log_path)

    return logger


def str_to_list(arg: str = '10-7-5-4-3'):
    '''
    Turn a string of numbers into a list.

    Inputs:
      arg: (str) must be of the form like "1-2-3-4".

    Returns:
      (list) example: [1, 2, 3, 4].
    '''

    return [int(i) for i in arg.strip().split('-')]


# #########################################################################
# 2. Helper for network and model training
# #########################################################################
def act_dict(act_name: str = 'tanh'):
    """
    Get a nn activation layer with a string input.
    This will later be used in files in ../.network/*.py.
    """
    assert act_name in ['relu', 'softmax', 'tanh', 'sigmoid', 'identity']

    dict_ = {'relu': nn.ReLU(),
             'softmax': nn.Softmax(),
             'tanh': nn.Tanh(),
             'sigmoid': nn.Sigmoid(),
             'identity': nn.Identity()}

    return dict_[act_name]


def loss_dict(loss_name: str = 'ce'):
    dict_ = {'ce': nn.CrossEntropyLoss()}

    return dict_[loss_name]


def get_module_by_name(net, access_string='cnn_model.0'):
    """
    Inputs:
        net: (nn.Module)
        access_string: (str) the name of the module
    Returns:
        (torch.nn.modules.conv/linear/...)
    """
    names = access_string.split(sep='.')
    return reduce(getattr, names, net)


# =========================================================================
# Helper for initialization
# =========================================================================
def init_weights_xavier(net):
    """
    Initialize with Xavier's method.
    """
    for module in net.named_modules():
        if isinstance(module[1], (nn.Linear, nn.Conv2d)):
            # Initialize for weights
            try:
                torch.nn.init.xavier_normal_(module[1].weight_orig)
            except:
                torch.nn.init.xavier_normal_(module[1].weight)

            # Initialize for bias
            if module[1].bias is not None:
                module[1].bias.data.fill_(0.01)


def init_weights_kaiming(net):
    """
    Initialize with Kaiming's method.
    """

    # ---------- Last layer: normal init to keep variance --------- #
    # This is actually foolish
    # Useless when the last linear layer is followed by a softmax layer
    module_last = [module for module in net.modules()][-1]

    if isinstance(module_last, nn.Linear):
        # Initialize for bias
        if module_last.bias is not None:
            nn.init.constant_(module_last.bias, 0)

        # Initialize for weights
        try:
            nn.init.normal_(module_last.weight_orig, 0, 0.01)
        except:
            nn.init.normal_(module_last.weight, 0, 0.01)

    # ------------- All other layers: use Kaiming init ------------ #
    module_list = [module for module in net.modules()][:-1]

    for module in module_list:
        # [Linear and Conv2d]: use Kaiming init
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Initialize for bias
            if module.bias is not None:
                module.bias.data.zero_()

            # Initialize for weights
            try:
                nn.init.kaiming_normal_(module.weight_orig,
                                        mode='fan_out',
                                        nonlinearity='relu')
            except:
                nn.init.kaiming_normal_(module.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

        # [BatchNorm]: directl fill with 1 and 0
        elif isinstance(module, nn.BatchNorm2d):
            # Initialize for bias
            if module.bias is not None:
                module.bias.data.zero_()

            # Initialize for weights
            try:
                if module.weight_orig is not None:
                    module.weight_orig.data.fill_(1.0)
            except:
                if module.weight is not None:
                    module.weight.data.fill_(1.0)


# =========================================================================
# Get layer output in order to calculate MI per layer
# =========================================================================
def get_layer_output(module_name, layer_outputs_dict):
    """
    This is used to get the output from each layer.
    You should be expected to use the function as follows:
    ```
    layer_outputs_dict = {}
    for module_name, layer in net.named_modules():
        layer.register_forward_hook(get_layer_outputs(module_name,
                                                      layer_outputs_dict))
    ```
    """

    def hook(model, inputs, outputs):
        layer_outputs_dict[module_name] = outputs.detach()

    return hook


# =========================================================================
# One-line set up for network in experiment-fisher.py
# =========================================================================
def setup_network(net_name, in_dim, out_dim, hidden_act,
                  out_act, hidden_dims, depth, widen_factor,
                  dropRate, growthRate, compressionRate,
                  mode, prune_method, prune_ratio, device,
                  prune_last=False, parallel=True):
    """
    This function will be used in experiment.py to shorten code.
    """
    # Build up the network
    net = build_network(net_name, in_dim, out_dim, hidden_act,
                        out_act, hidden_dims, depth, widen_factor,
                        dropRate, growthRate, compressionRate)

    # Set the network in case it is a pruned one (for convenience of loading)
    if mode in ['sparse-finetune', 'sparse-scratch', 'lottery']:
        # This will simply add weight_orig and weight_mask to state_dict
        global_prune(net, prune_method, prune_ratio, prune_last)

    # Make assertion on the device
    assert device in ['cuda', 'cpu'], 'TPU is not supported in experiments.'

    # Load the network to the device
    if parallel:
        net = torch.nn.DataParallel(net).to(device)
    else:
        net = net.to(device)

    return net


# #########################################################################
# 3. Helper for optimizer
# #########################################################################
# =========================================================================
# Get accuracy in a batch
# =========================================================================
def get_acc(outputs, y, batch_size):
    """
    Get the accuracy for a batch of predictions.

    Inputs:
        outputs: (torch.tensor) the predicted label distributions
                 shape=(batch_size, n_classes)
        y: (torch.tensor) the true labels
           shape=(batch_size,)
        batch_size: (int) as the name suggested

    Return:
        The accuracy for this prediction.
    """
    acc = (outputs.argmax(dim=1) == y).type(torch.float).sum()
    acc /= batch_size
    return acc.detach()


# =========================================================================
# Get current memory
# =========================================================================
def get_memory():
    """
    Get current memory in GB.
    """
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3


# =========================================================================
# Calculating Moving Averages
# =========================================================================
class Tracker():
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.mean = 0
        self.sum = 0
        self.count = 0

    def update(self, val, batch_size=1):
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        self.mean = self.sum / self.count


# =========================================================================
# Calculating Moving Averages
# =========================================================================
class Welford(object):
    """
    Implements Welford's algorithm for computing a running mean
    and standard deviation as described at:
        http://www.johndcook.com/standard_deviation.html
    can take single values or iterables
    Properties:
        mean    - returns the mean
        std     - returns the std
        meanfull- returns the mean and std of the mean
    Usage:
        >>> foo = Welford()
        >>> foo(range(100))
        >>> foo
        <Welford: 49.5 +- 29.0114919759>
        >>> foo([1]*1000)
        >>> foo
        <Welford: 5.40909090909 +- 16.4437417146>
        >>> foo.mean
        5.409090909090906
    """

    def __init__(self, lst=None):
        self.k = 0
        self.M = 0
        self.S = 0

        self.__call__(lst)

    def update(self, x):
        if x is None:
            return
        self.k += 1
        newM = self.M + (x - self.M) * 1. / self.k
        newS = self.S + (x - self.M) * (x - newM)
        self.M, self.S = newM, newS

    def consume(self, lst):
        lst = iter(lst)
        for x in lst:
            self.update(x)

    def __call__(self, x):
        if hasattr(x, "__iter__"):
            self.consume(x)
        else:
            self.update(x)

    @property
    def mean(self):
        return self.M

    @property
    def meanfull(self):
        return self.mean, self.std / torch.sqrt(self.k)

    @property
    def std(self):
        if self.k == 1:
            return 0
        return torch.sqrt(self.S / (self.k - 1))

    def __repr__(self):
        return "<Welford: {} +- {}>".format(self.mean, self.std)


class WelfordBs(object):
    """
    Batch-wise welford algorithm.
    """

    def __init__(self, lst=None):
        self.k = 0
        self.M = 0
        self.S = 0

        self.__call__(lst)

    def update(self, x):
        if x is None:
            return
        self.k += 1
        newM = self.M + (x - self.M) * 1. / self.k
        newS = self.S + (x - self.M) * (x - newM)
        self.M, self.S = newM, newS

    def __call__(self, x):
        self.update(x)

    @property
    def mean(self):
        return self.M

    @property
    def meanfull(self):
        return self.mean, self.std / torch.sqrt(self.k)

    @property
    def std(self):
        if self.k == 1:
            return 0
        return torch.sqrt(self.S / (self.k - 1))


# =========================================================================
# Save state dict
# =========================================================================
# def get_net_dict(net, device):
#     """
#     This unnecessary function shows you how unfriendly Torch is on TPU.
#     """
#     # On non-TPU devices
#     if device in ['cuda', 'cpu']:
#         try:
#             # Handle the case when nn.DataParallel() wraps a module
#             net_dict = net.module.state_dict()
#         except:
#             net_dict = net.state_dict()
#
#     # On TPU devices
#     else: # It should be set outside as xm.xla_device()
#         # Move net to CPU for state_dict so that it can be loaded outside TPU
#         net_dict = net.cpu().state_dict()
#         # Move back net in order it can be trained
#         net = xmp.MpModelWrapper(net)
#         net = net.to(device=device)
#
#     return net, net_dict

def load_state_dict_(net, state_dict):
    """
    A helper function to handle the loading issue.
    """
    try:
        net.load_state_dict(state_dict)
    except:
        try:
            net.module.load_state_dict(state_dict)
        except:
            for key in list(state_dict.keys()):
                state_dict[key.replace('module.', '')] = state_dict.pop(key)
            net.load_state_dict(state_dict)
    return net


def get_net_dict(net, device):
    """
    This unnecessary function shows you how unfriendly Torch is on TPU.
    """
    try:
        # Handle the case when nn.DataParallel() wraps a module
        net_dict = net.module.state_dict()
    except:
        net_dict = net.state_dict()

    return net_dict


def save_net_dict(net_dict,
                  epoch: int = 0,
                  is_last_epoch: bool = False,
                  path: str = 'final_path/state_dicts/epoch_0.pkl'):
    """
    Save essential information according to the epoch number.
    """
    try:
        save_func = xm.save
    except:
        save_func = torch.save

    if epoch <= 50 and not epoch % 2:
        save_func(net_dict, path)

    elif epoch <= 100 and not epoch % 5:
        save_func(net_dict, path)

    elif epoch <= 200 and not epoch % 10:
        save_func(net_dict, path)

    elif epoch <= 500 and not epoch % 30:
        save_func(net_dict, path)

    elif epoch <= 2000 and not epoch % 50:
        save_func(net_dict, path)

    elif epoch <= 5000 and not epoch % 80:
        save_func(net_dict, path)

    elif epoch > 5000 and not epoch % 120:
        save_func(net_dict, path)

    elif is_last_epoch:
        save_func(net_dict, path)

    return None


def get_saved_epoch_list(n_epoch):
    """
    This is a silly function.
    Just to get a list of epoch id list.
    This will be used in experiments to decide which to load.
    """

    # A helper function on if this epoch saves
    # def get_saved_epoch(epoch):
    #     if epoch <= 100 and not epoch % 10:
    #         return epoch
    #
    #     elif epoch <= 200 and not epoch % 20:
    #         return epoch
    #
    #     elif epoch <= 500 and not epoch % 30:
    #         return epoch
    #
    #     elif epoch <= 2000 and not epoch % 50:
    #         return epoch
    #
    #     elif epoch <= 5000 and not epoch % 80:
    #         return epoch
    #
    #     return None

    def get_saved_epoch(epoch):
        if epoch <= 50:
            return epoch

        elif epoch <= 100 and not epoch % 5:
            return epoch

        elif epoch <= 200 and not epoch % 10:
            return epoch

        elif epoch <= 500 and not epoch % 30:
            return epoch

        elif epoch <= 2000 and not epoch % 50:
            return epoch

        elif epoch <= 5000 and not epoch % 80:
            return epoch

        return None

    # Add the saved epochs to the list
    saved_epoch_list = []
    for epoch in range(n_epoch):
        if get_saved_epoch(epoch) is not None:
            saved_epoch_list.append(get_saved_epoch(epoch))

    # Make sure you do not miss the last epoch
    if n_epoch not in saved_epoch_list:
        saved_epoch_list.append(n_epoch)

    return saved_epoch_list


def save_net_dicts_tpu(net, final_path, n_epochs, device):
    """
    Resave the net dicts in CPU.
    We cannot do this in training as the optimizer will hate us.

    Input:
        net (nn.Module), final_path (path.Path)
    Returns:
        None. The net dicts will be resaved.
    """

    # Make an anouncement
    print('I, the machine, am horsing around with TPU.')

    # Get the list of net dicts' path
    net_dict_list = get_saved_epoch_list(n_epochs)
    net_dict_paths = [final_path / 'state_dicts' / f'epoch_{i}.pkl'
                      for i in net_dict_list]
    net_dict_paths.append(final_path / 'model.tar')
    net_dict_paths.append(final_path / 'model_best.tar')

    # Re-save the net dicts in CPU format
    # We cannot do this in training as optimizer will hate that
    for net_dict_path in net_dict_paths:
        # Load the state dict
        try:
            net_dict = torch.load(net_dict_path, map_location='cpu')
        except:
            pass

        # Move net to CPU
        net = net.cpu()
        net.load_state_dict(net_dict)
        net_dict = net.cpu().state_dict()

        # Save the CPU version net dict
        torch.save(net_dict, net_dict_path)

        # Move back net from CPU to TPU for next iter
        net = xmp.MpModelWrapper(net)
        net = net.to(device=device)

    return None


# =========================================================================
# 4. Calculating gradients
# =========================================================================
def get_snr(net,
            data_loader,
            criterion,
            optimizer,
            grad_dict,
            epoch,
            device,
            final_path,
            prune_indicator):
    """
    This function with save the mean/std for gradients and weights of each layer.
    We first record the moving mean/std, then calculate the norm of the moving mean/std.

    The detailed description for the arguments can be found in ../optim/trainer.py.
    """

    # ------------ Set up an epoch-wise dictionary ------------- #
    # Set up the layer dict, where the key is the layer name
    # Notice that this is a transient dict
    # The information will be extracted and then saved in grad_dict
    layer_dict = OrderedDict()
    for name, layer in net.named_modules():
        if type(layer) in (nn.Linear, nn.Conv2d):
            layer_dict[name] = {'weight_welford': WelfordBs(),
                                'grad_welford': WelfordBs()}

    # ------------ Get statistics for (moving) gradient mean ------------- #
    for data in data_loader:
        # Load data
        inputs, y, _ = data
        if device in ['cpu', 'cuda']:
            inputs, y = inputs.to(device), y.to(device)

        # Calculate gradients
        outputs = net(inputs)
        loss = criterion(outputs, y)
        # optimizer.zero_grad()  # Should better use net.zero_grad()
        net.zero_grad()
        loss.backward()

        # Record gradients; appending n_batches_grad per epoch
        for name, layer in net.named_modules():
            if type(layer) in (nn.Linear, nn.Conv2d):
                # Get gradients according to pruning or not
                if prune_indicator:
                    weight_mask_ = layer.weight_mask.detach().reshape(-1).bool()
                    grad_ = layer.weight_orig.grad.detach().reshape(-1)[weight_mask_]
                    weight_ = layer.weight.detach().reshape(-1)[weight_mask_]
                else:
                    grad_ = layer.weight.grad.detach().reshape(-1)
                    weight_ = layer.weight.detach().reshape(-1)

                # Get the moving average
                layer_dict[name]['grad_welford'](grad_)
                layer_dict[name]['weight_welford'](torch.norm(weight_))

                # Debug note
                # The calculating for weights seems to be redundant
                # We use the same weights (network) during this process

        # optimizer.zero_grad()
        net.zero_grad()
        del grad_, weight_

    # ------------ Get mean of epoch-wise gradients ------------- #
    # Get essential data to draw SNR plot
    for name, layer in net.named_modules():
        if type(layer) in (nn.Linear, nn.Conv2d):
            # Get gradient mean and its norm
            grad_welford_mean_norm = torch.norm(layer_dict[name]['grad_welford'].mean)
            grad_dict[f'epoch {epoch}']['grad_mean'].append(grad_welford_mean_norm)

            # Get gradient std
            grad_welford_std_norm = torch.norm(layer_dict[name]['grad_welford'].std)
            grad_dict[f'epoch {epoch}']['grad_std'].append(grad_welford_std_norm)

            # Get weight norm
            weight_welford_norm = layer_dict[name]['weight_welford'].mean
            grad_dict[f'epoch {epoch}']['weight_norm'].append(weight_welford_norm)

    # ------------ Save results ------------- #
    torch.save(grad_dict, final_path / 'grad_dict.pkl')
    return None


# =========================================================================
# 5. Calculating Fisher Information (using nngeometry package)
# =========================================================================
def get_fisher(net,
               data_loader,
               representation,
               fisher_metric: str = 'fim_monte_carlo',
               monte_carlo_trials: int = 1,
               n_output: int = 10,
               variant: str = 'classif_logits',
               device: str = 'cuda',
               final_path: str = './final_path'):
    """
    This is a simple implementation to calculate the trace of fisher information.
    Reference: github.com/tfjgeorge/nngeometry

    Inputs:
        net: (nn.Module) the network to get the fisher information
        data_loader: (DataLoader) the loader of the dataset
        representation: (object) we will use PMatEKFAC as default, see trainer.py
        fisher_metric: (str) use fim (the short for fisher information matrix)
                             for closed form solution, and fim_monte_carlo for
                             estimated solution; the latter is faster, presumably.
        monte_carlo_trials: (int) the trials used for fim_monte_carlo; 5 is default.
        The rest arguments are default or can be automatically set (just lazy).
        The final path is redundant, but let us keep it for now.

    Returns:
        The trace of the fisher matrix.
        Ideally we can compute more, but for now let's stick with trace.
    """
    # Handle the case when the input net is already wrapped by nn.DataParallel()
    try:
        net = net.module  # nn.DataParallel() will add a module outside the net
    except:
        pass

    # Calculate only linear and Conv2d layers
    layer_collection = LayerCollection()
    for layer in net.modules():
        if type(layer) in (nn.Linear, nn.Conv2d):
            layer_collection.add_layer_from_model(net, layer)

    # Generate a K-FAC representation
    if fisher_metric == 'fim':
        F_info = FIM(layer_collection=layer_collection,
                     model=net,
                     loader=data_loader,
                     representation=representation,
                     n_output=n_output,
                     variant=variant,
                     device=device)

    elif fisher_metric == 'fim_monte_carlo':
        F_info = FIM_MonteCarlo(layer_collection=layer_collection,
                                model=net,
                                loader=data_loader,
                                representation=representation,
                                variant='classif_logits',
                                trials=monte_carlo_trials,
                                device=device)

    return F_info.trace().item()


# =========================================================================
# Estimating Trace of Fisher Information (a Naive Implementation)
# =========================================================================
def get_fisher_ours(net,
                    data_loader,
                    n_samples: int = 5,
                    n_trials: int = 5,
                    device: str = 'cuda'):
    """
    Estimate the trace of (true) fisher information by getting
    the expectation of norms on gradients from predicted y.

    Inputs:
        net (nn.Module): The network to get gradient.
        data_loader (DataLoader): The dataset to evaluate fisher.
            Be aware that batch size must be 1.
        n_samples (int): The number of monte carlo samples of X.
        n_trials (int): Given an X, the number of monte carlo samples
            of its predicted label.

    Returns:
        The (approx) trace of (true) fisher information.
    """
    # Set network in eval mode
    net.eval()

    # Set the recorder for running mean of fisher trace
    trace_recorder = WelfordBs()

    # We use cross-entropy for classification problems
    criterion = nn.CrossEntropyLoss()

    loss_tracker = WelfordBs()  # debug

    # Start calculating fisher trace
    for n_iter, data in enumerate(data_loader):
        # Only sample for designated numbers; shuffle must set to be True.
        if n_iter >= n_samples: break
        # It seems that we evaluate fisher on the same samples of data
        # every time we call this function. Is this correct?

        # Get values from the loader; batch size must be 1
        inputs = data[0].to(device)

        # Get the predicted distribution and use MC sampling for predicted label
        with torch.no_grad():
            outputs = net(inputs)
            probs = torch.softmax(outputs.detach(), dim=1)
            sampled_y_pred = torch.multinomial(probs, n_trials, replacement=True)
            sampled_y_pred.resize_(n_trials, 1)

        # Set the recorder for running mean arising from sampling y
        trace_recorder_inside = WelfordBs()

        # Sample several predicted labels
        for y_pred in sampled_y_pred:

            # Zero out the gradients
            net.zero_grad()

            # Get the outputs
            outputs = net(inputs)  # This is unnecessary, just to debug

            # Get the gradient evaluated with predicted y
            loss = criterion(outputs, y_pred.detach())
            loss_tracker(loss)  # just to debug
            net.zero_grad()  # Just to debug

            # Backpropagate for loss
            loss.backward()

            # Initialize the fisher trace
            fisher_trace = 0

            # Calculate fisher trace by getting squared norm of grads
            for name, module in net.named_modules():

                # Only calculate the gradients from Conv and Linear layers
                if not type(module) in (nn.Linear, nn.Conv2d):
                    continue

                # Get the gradients when network is pruned
                try:
                    # Get the pruning mask
                    mask = module.weight_mask.detach()
                    # Applying the mask to the gradients
                    grad = module.weight_orig.grad.detach()
                    grad[~mask.bool()] = 0

                # Get the gradients for usual network
                except:
                    grad = module.weight.grad.detach()

                # Calculate the trace
                fisher_trace += torch.norm(grad) ** 2

            # Record the running mean from sampling y
            trace_recorder_inside(fisher_trace.detach())

            del loss, outputs  # This is unnecessary, just to debug

        # Save the running mean (to approx expectation) of trace
        trace_recorder(trace_recorder_inside.mean)
    print(f'Loss mean on sampled label: {loss_tracker.mean.item():.3f}', )

    return trace_recorder.mean.item()


# =========================================================================
# Estimating Trace of Fisher Information (using vmap for efficiency)
# =========================================================================
def get_fisher_vmap(net,
                    data_loader,
                    n_samples: int = 10000,
                    n_trials: int = 5,
                    batch_size: int = 500,
                    param_name_list: tuple = ('conv1.weight',),
                    mode: str = 'sparse-scratch',
                    mask_dict: dict = {},
                    device: str = 'cuda'):
    """
    WARNING: STILL IN DEVELOPMENT.
    An efficient way to calculate fisher info by functorch.vmap.
    Batch size at most 600 for P100 GPU machines.
    """
    # --------------- Preparations --------------- #
    # WARNING: be sure that pruning has been removed for net
    # Or vmap will be unsuccessful
    # Set network in eval mode
    net.eval()
    replace_all_batch_norm_modules_(net)  # handle vmap issue

    # Set the recorder for running mean of fisher trace
    trace_recorder = WelfordBs()

    # We use cross-entropy for classification problems
    criterion = nn.CrossEntropyLoss()

    # Get the maximum number of iterations
    n_iters = n_samples // batch_size

    for iter_i, data in enumerate(data_loader):
        # Set the break condition
        if iter_i > n_iters:
            break

        # Set the trace_recorder for this iteration
        trace_recorder_inside = WelfordBs()

        # Get the inputs
        inputs = data[0].to(device)

        # Zero out the gradients
        net.zero_grad()

        # Separate the state (i.e. parameters) from the net
        # func_net is now a pure stateless function
        func_net, params, buffers = make_functional_with_buffers(net)

        # --------------- Get predicted labels --------------- #
        # Helper function for get y_pred per sample
        def y_pred_per_sample(params, buffers, sample):
            """
            Sample the predicted targets.

            Returns:
                (torch.tensor) shape = (n_trials, 1)
            """

            # Manually add the batch dimension for the per sample setting
            sample = sample.unsqueeze(0)

            # Replace `prediction = model(sample)` with stateless models.
            # **Global variable**: `func_net`
            with torch.no_grad():
                output = func_net(params, buffers, sample)
                probs = torch.softmax(output.detach(), dim=1)
                sampled_y_pred = torch.multinomial(probs, n_trials, replacement=True)
                sampled_y_pred.resize_(n_trials)

            return sampled_y_pred

        # Get the vmap function for efficient calculation
        y_pred_vmap = vmap(y_pred_per_sample, in_dims=(None, None, 0), randomness='different')

        # Get n_trials of predicted labels for each sample
        y_preds = y_pred_vmap(params, buffers, inputs)  # shape = (batch_size, n_trials, 1)

        # Start calculating
        for trial_i in range(n_trials):

            # Extract the predicted label for this trial
            y_pred = y_preds[:, trial_i]  # shape = (batch_size,)

            # Zero out the gradients
            net.zero_grad()

            # Get func_net, params and buffer again (may be silly)
            func_net, params, buffers = make_functional_with_buffers(net)

            # --------------- Start: Vmap to get gradients --------------- #
            # Helper function to get loss per sample
            def loss_per_sample(params, buffers, sample, target):
                """
                Compute gradient for one sample.

                Inputs:
                    One sample without the batch size dimension.

                Returns:
                    A tuple, where each element is a parameter like weight
                    or bias of a certain layer.
                """

                # Manually add the batch dimension for the per sample setting
                sample = sample.unsqueeze(0)
                target = target.unsqueeze(0)

                # Replace `prediction = model(sample)` with stateless models.
                # **Global variable**: `func_net`, a fixed stateless function
                output = func_net(params, buffers, sample)

                # Loss function is the same as before
                loss = criterion(output, target)
                return loss

            # Get the gradient function from the loss function
            grad_per_sample = grad(loss_per_sample)

            # Apply vmap on the gradient function
            grad_vmap = vmap(grad_per_sample, in_dims=(None, None, 0, 0))
            # --------------- End: Vmap to get gradients --------------- #

            # Get per sample gradients into a tuple
            grads = grad_vmap(params, buffers, inputs, y_pred.detach())

            # Initialize the placeholder for fisher trace
            fisher_trace = 0

            # Accumulate fisher trace by grad norm
            for ind, name in enumerate(param_name_list):
                # Condition 1: only get gradients for weights or bias
                if not name.endswith(('.weight', '.bias')):
                    continue

                # Condition 2: only get gradients for linear and conv2d
                try:
                    module_name = name.replace('.weight', '')
                except:
                    module_name = name.replace('.bias', '')

                # Get the module
                module = get_module_by_name(net, module_name)
                if not type(module) in (nn.Linear, nn.Conv2d):
                    continue

                # Get the gradients for the bias without mask
                if name.endswith('bias'):
                    with torch.no_grad():
                        fisher_trace += torch.norm(grads[ind]) ** 2
                    # No need to apply mask for bias layer
                    continue

                # Mask the weight gradients for prunned networks
                if mode in ['sparse-finetune', 'sparse-scratch', 'lottery']:
                    # Masked out the pruned layer
                    try:
                        mask = mask_dict[module_name + '.weight_mask']

                        # Helper function for vmap mask
                        def get_masked_per_sample(sample):
                            return sample * mask

                        # Apply vmap on the gradient function
                        masked_vmap = vmap(get_masked_per_sample, in_dims=(0))

                        # Replace the gradients with the masked one
                        with torch.no_grad():
                            masked_grad = masked_vmap(grads[ind])
                            grads[ind] = masked_grad
                    # Handle the case when certain layers are not pruned
                    except:
                        pass

                # Get the fisher norm for this module (it's total norm of batches)
                with torch.no_grad():
                    fisher_trace += torch.norm(grads[ind]) ** 2

            # Get the mean fisher trace in this batch
            fisher_trace /= inputs.shape[0]

            # Record the fisher trace to get running mean
            trace_recorder_inside(fisher_trace.detach())

        # Get the running mean in each sampled batch
        trace_recorder(trace_recorder_inside.mean)

    return trace_recorder.mean.item()


# #########################################################################
# 5. Helper for Linear Interpolation
# #########################################################################
def sort_dict(dict_):
    """
    Sort the dictionary by value.
    """
    return dict(sorted(dict_.items(), key=lambda item: item[1]))


def compute_loss(net, data_loader, criterion, device='cuda', train=False):
    """
    Get the loss for one round of computation.
    """
    if train:
        net.train()
    else:
        net.eval()

    with torch.no_grad():
        loss_all, total = 0, 0

        for ind, data in enumerate(data_loader):
            # Get the data
            inputs, y, _ = data
            inputs, y = inputs.to(device), y.to(device)

            # Get outputs
            outputs = net(inputs)

            # Calculate loss
            losses_train = criterion(outputs, y)

            # Update statistics
            n_data = inputs.shape[0]
            total += n_data
            loss_all += losses_train * n_data

    # Just in case
    net.zero_grad()

    return (loss_all / total).detach().item()


def warm_bn(net, data_loader, device):
    """
    Update running statistics for batch norm layers.
    This is used after the interpolation of batch norm layers,
    as we cannot directly interpolate running mean/var.
    """
    # Reset running statistics
    for module in net.modules():
        if isinstance(module, (nn.BatchNorm2d,)):
            module.reset_running_stats()

    # Record and refresh the training mode
    training = net.training
    net.train()

    # Update the running statistics
    with torch.no_grad():
        for data in data_loader:
            # Get the data
            inputs, y, _ = data
            inputs, y = inputs.to(device), y.to(device)
            outputs = net(inputs)

    # Set back the training mode
    net.train(training)


def get_direction(net_inter,
                  net_finetune,
                  param_name_list,
                  mask_dict):
    """
    Deprecated! We now save by tensor (see the next function)
    instead of by dictionary.

    Get the direction from net_finetune to net_inter.

    The results are saved in dict, where the keys matches
    the keys from net.named_paramerters() yet only for those
    linear and conv2d layers with weights.
    """

    # Initialize the direction dict
    direction_dict = dict()

    # Save the direction per layer
    for ind, name in enumerate(param_name_list):
        # Only consider the direction for weights
        if not '.weight' in name:
            continue

        # Get the module
        name = name.replace('.weight', '')
        module_inter = get_module_by_name(net_inter, name)
        module_finetune = get_module_by_name(net_finetune, name)

        # Only consider weight for linear and conv2d layers
        if not type(module_inter) in (nn.Linear, nn.Conv2d):
            continue

        # Save the direction for this layer
        with torch.no_grad():
            # Get the direction
            direction = module_finetune.weight.detach() - module_inter.weight.detach()

            # Masked out the pruned layer
            try:
                mask = mask_dict[name + '.weight_mask']
                direction *= mask

            # Handle the case when certain layers are not pruned
            except:
                pass

            # Save the direction
            direction_dict[name] = direction

    # Just in case
    net_inter.zero_grad()
    net_finetune.zero_grad()

    return direction_dict


def get_direction_vector(net_inter: nn.Module,
                         net_finetune: nn.Module):
    """
    Inputs:
        net_inter: (nn.Module) the network being interpolated
        net_finetune: (nn.Module) the pruned and finetuned network

    Returns:
        direction_vector: (torch.Tensor) shape = (n_parameters,)
    """
    # Get the parameters for the finetuned
    net_finetune_params = [param for param in net_finetune.parameters()]

    # Initialize to be stacked later
    direction_vector = []

    # Get direction per layer
    for i, param_inter_i in enumerate(net_inter.parameters()):
        direction_i = (net_finetune_params[i] - param_inter_i).reshape(-1)
        direction_vector.append(direction_i)

    # Concatenate to get a long vector
    direction_vector = torch.cat(direction_vector)

    return direction_vector


def get_cos_dict(net,
                 data_loader,
                 param_name_list: tuple = ('conv1.weight',),
                 mask_dict: dict = {},
                 direction_vector: torch.Tensor = (),
                 device: str = 'cuda'):
    """
    Get the gradients for each sample in the training set.
    """
    # Set network in eval mode
    # Make sure that the interpolated network has updated the running
    # stats, i.e., going over a train mode before
    net.eval()
    replace_all_batch_norm_modules_(net)  # handle vmap issue

    # Initialize cos dict, key is ind and value is cos to direction
    cos = nn.CosineSimilarity(dim=0)
    cos_dict = dict()

    # Set the default criterion
    criterion = nn.CrossEntropyLoss()

    # Start getting gradients
    for debug_i, data in enumerate(data_loader):
        # Debug
        start_time = time.time()

        # Get the data
        inputs, y, idx = data
        inputs, y = inputs.to(device), y.to(device)
        idx = idx.to(device)

        # Zero out the gradients
        net.zero_grad()

        # Get func_net, params and buffer again (may be silly)
        func_net, params, buffers = make_functional_with_buffers(net)

        # --------------- Vmap to get gradients --------------- #
        # Helper function to get loss per sample
        def loss_per_sample(params, buffers, sample, target):
            """
            Compute gradient for one sample.

            Inputs:
                One sample without the batch size dimension.

            Returns:
                A tuple, where each element is a parameter like weight
                or bias of a certain layer.
            """

            # Manually add the batch dimension for the per sample setting
            sample = sample.unsqueeze(0)
            target = target.unsqueeze(0)

            # Replace `prediction = model(sample)` with stateless models.
            # **Global variable**: `func_net`, a fixed stateless function
            output = func_net(params, buffers, sample)

            # Loss function is the same as before
            loss = criterion(output, target)
            return loss

        # Get the gradient function from the loss function
        grad_per_sample = grad(loss_per_sample)

        # Apply vmap on the gradient function
        grad_vmap = vmap(grad_per_sample, in_dims=(None, None, 0, 0))

        # Get per sample gradients into a tuple
        grads = grad_vmap(params, buffers, inputs, y)

        # --------------- Mask the gradients --------------- #
        for ind, name in enumerate(param_name_list):
            # Condition 1: only apply mask for weights
            if '.weight' not in name:
                continue

            # Condition 2: only apply mask for linear and conv2d
            name = name.replace('.weight', '')
            module = get_module_by_name(net, name)
            if not type(module) in (nn.Linear, nn.Conv2d):
                continue

            # Masked out the pruned layer
            try:
                mask = mask_dict[name + '.weight_mask']

                # Helper function for vmap mask
                def get_masked_per_sample(sample):
                    return sample * mask

                # Apply vmap on the gradient function
                masked_vmap = vmap(get_masked_per_sample, in_dims=(0))

                # Get masked_grad; shape = (batch_size, (weight's shape))
                with torch.no_grad():
                    masked_grad = masked_vmap(grads[ind])

                # Replace the grad to be the masked ones
                grads[ind] = masked_grad

            # Handle the case when certain layers are not pruned
            except:
                pass

        # --------------- Get the cosine similarity --------------- #
        # Reshape the tuple grads, whose length is n_parameters
        grads_stack = grads[0].reshape(grads[0].shape[0], -1)
        for grad_i in grads[1:]:
            grad_i = grad_i.reshape(grad_i.shape[0], -1)
            grads_stack = torch.hstack((grads_stack, grad_i))

        # Helper function to get vmap on geting cos to dir
        def get_cos_per_sample(sample):
            return cos(direction_vector, - sample)

        # Apply vmap to get cos similarity function
        cos_vmap = vmap(get_cos_per_sample, in_dims=(0))

        # Get cos results, shape = (batch_size,)
        with torch.no_grad():
            cos_results = cos_vmap(grads_stack)

        # Save the results in dict
        for i, idx_i in enumerate(idx):
            cos_dict[idx_i] = cos_results[i]

        # Debug
        if not debug_i % 50:
            print(f'One round time at {debug_i}: {time.time() - start_time:.3f}s ')

    # Clear out the gradients
    net.zero_grad()

    return sort_dict(cos_dict)
