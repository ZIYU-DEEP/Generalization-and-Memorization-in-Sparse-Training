"""
Title: loader_base.py
Description: The base trainer and evaluater.
"""

from abc import ABC, abstractmethod


# #########################################################################
# 1. Base Loader
# #########################################################################
class BaseLoader(ABC):
    def __init__(self):
        super().__init__()
        self.train_set = None  # must be of type torch.helper.data.Dataset
        self.test_set = None  # must be of type torch.helper.data.Dataset
        self.all_set = None  # must be of type torch.helper.data.Dataset

    @abstractmethod
    def loaders(self,
                batch_size: int,
                shuffle_train: bool=True,
                shuffle_test: bool=False,
                shuffle_all: bool=False,
                num_workers: int=0):
        """
        Implement data loaders of type torch.helper.data.DataLoader.
        """
        pass

    def __repr__(self):
        return self.__class__.__name__
