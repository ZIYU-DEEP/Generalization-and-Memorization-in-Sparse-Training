"""
[Title] pruner.py
[Description] The simplest prune schedule by PyTorch.
"""

from torch import nn
from functools import reduce
import torch.nn.utils.prune as torch_prune


def global_prune(net: nn.Module,
                 prune_method: str='l1',
                 prune_ratio: float=0.6,
                 prune_last: bool=False):
    """
    This is the simplest pruning for network.
    Note that we use default L1 unstructure pruning.

    Inputs:
        net: (nn.Module) the net to be pruned
        prune_method: (str) the method for pruning, currently support l1
        prune_ratio: (float) the ratio of weights to be pruned

    Return:
        None (pruning will take effect in place)
    """
    # Get the module list
    module_list = [module for module in net.modules()]

    # We usually don't prune the last layer
    if not prune_last:
        module_list = module_list[:-1]


    # Prune only the dense layers and Conv2d layers
    parameters_to_prune = []
    for module in module_list:
        if type(module) in (nn.Linear, nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))

    if prune_method == 'l1':
        torch_prune.global_unstructured(parameters_to_prune,
                                        pruning_method=torch_prune.L1Unstructured,
                                        amount=prune_ratio)

    if prune_method == 'random':
        torch_prune.global_unstructured(parameters_to_prune,
                                        pruning_method=torch_prune.RandomUnstructured,
                                        amount=prune_ratio)

    return None


def global_prune_custom_mask(net: nn.Module,
                             pretrain_net: nn.Module):
    """
    The goal is to use the mask from a pretrained network to prune a new net.
    That is, we apply the weight mask from the pruned pretrained network on our net.
    The net to be pruned and the pretrained net should have the same structure.

    Inputs:
        net: (nn.Module) the net to be pruned
        pretrain_net: (nn.Module) the other pretrained net

    Return:
        None (pruning will take effect in place)
    """
    # Get the module list (we don't prune the last layer)
    module_list = [module for module in net.modules()][:-1]
    pre_module_list = [module for module in pretrain_net.modules()][:-1]

    # Prune the net with the mask from pretrain_net
    for module, pre_module in zip(module_list, pre_module_list):
        if type(module) in (nn.Linear, nn.Conv2d):
            mask = pre_module.weight_mask
            torch_prune.custom_from_mask(module, 'weight', mask)

    return None


def get_module_by_name(net, access_string='cnn_model.0'):
    """
    This should really be imported from utils but i am silly.
    Inputs:
        net: (nn.Module)
        access_string: (str) the name of the module
    Returns:
        (torch.nn.modules.conv/linear/...)
    """
    names = access_string.split(sep='.')
    return reduce(getattr, names, net)


def remove_prune(net: nn.Module):
    """
    Put weight back to be parameter, remove weight_orig and mask.
    Notice the the weight is still the pruned one.
    However the zero entries will not hold after gradient updates.
    """
    # Get the module list (we don't prune the last layer)
    module_list = [module for module in net.modules()][:-1]

    for module in module_list:
        if type(module) in (nn.Linear, nn.Conv2d):
            torch_prune.remove(module, 'weight')


def get_mask_dict(net: nn.Module):
    """
    Return a dict which saves the mask for each pruned module.
    The input must be a pruned network with weight_mask in attr.
    The masks are only collected for linear and conv2d layers.
    """
    # Initialize the mask dict
    mask_dict = dict()

    # Get the module list (we don't prune the last layer)
    name_module_list = [(name, module) for name, module
                        in net.named_modules()][:-1]

    # Save the mask
    for name, module in name_module_list:
        if type(module) in (nn.Linear, nn.Conv2d):
            name += '.weight_mask'
            mask_dict[name] = get_module_by_name(net, name)

    return mask_dict
