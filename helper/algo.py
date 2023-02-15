"""
[Title] algo.py
[Use] A helper file for training algorithms.
"""

from torch import nn

import torch
import torch.nn.functional as F



def top_k_idx(vec,
              k: int=128,
              largest: bool=True):
    """
    Returns the idx (indices) of the x largest/smallest entries in vec.

    Args:
        vec: tensor, number of samples to be selected
        k: int, number of idx to be returned
        largest: bool, if true, the k largest entries are selected; if false,
        the k smallest entries are selected
    Returns:
        top_k_idx: tensor, top x idx, sorted
        other_idx: tensor, the idx that were not selected
    """

    sorted_idx = torch.argsort(vec, descending=largest)

    top_k_idx = sorted_idx[:k]
    other_idx = sorted_idx[k:]

    return top_k_idx, other_idx


def subset_selection(inputs, y, net,
                     subset_ratio: float=0.1,
                     subset_method: str='loss'):
    """
    Select a subset from the current data batch.
    Supported methods:
        - default: the first k idx (use when data already shuffled)
        - uniform: uniformly randomly select (sgd)
        - loss: select with highest losses (ordered sgd)
        - fisher: select with highest fisher info
    """

    # Subset selection
    with torch.no_grad():

        # Set the size of the subset
        k = int(len(y) * subset_ratio)

        # Subset selection
        selected_idx = torch.arange(k)

        if subset_method == 'default':
            pass

        elif subset_method == 'uniform':
            selected_idx = torch.randperm(len(y))[:k]

        elif subset_method == 'loss':
            # Compute prediction error
            outputs = net(inputs)
            loss_pool = F.cross_entropy(outputs, y, reduction='none')

            # Get the selected idx
            selected_idx, _ = top_k_idx(loss_pool,
                                        k=k,
                                        largest=True)

        # Extract the inputs and y
        inputs, y = inputs[selected_idx], y[selected_idx]

    return inputs, y
