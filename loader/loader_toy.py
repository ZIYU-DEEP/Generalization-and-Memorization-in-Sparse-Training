"""
Title: loader_toy.py
Description: Loading pickled toy datasets.
"""

from .loader_base import BaseLoader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import os
import torch
import joblib


# #########################################################################
# 1. Dataset Class for Toy Datasets
# #########################################################################
class ToyDataset(Dataset):
    def __init__(self,
                 root: str='./data/',
                 filename: str='toy',
                 train: int=1,
                 test_size: float=0.2,
                 random_state: int=42):
        """
        This class should generally load any data in pickled format.
        You need to pickle data first with (X, y) tuple.
        Notice <train> is a indicator for loading method, see if condition below.
        """

        # Initialization
        self.path = root + f'{filename}.pkl'
        self.train = train
        self.test_size = test_size
        self.random_state = random_state
        self.n_classes = 2

        # Load data
        X, y = joblib.load(self.path)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state,
                                                            stratify=y)

        # Load designated data
        if not train:     # Only load test data
            X, y = X_test, y_test
        elif train == 1:  # Only load train data
            X, y = X_train, y_train
        elif train == 2:
            pass          # Load the full data

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int32)

    def __getitem__(self, index):
        sample, target = self.X[index], int(self.y[index])
        return sample, target, index

    def __len__(self):
        return len(self.X)


# #########################################################################
# 2. Loader for Toy Datasets
# #########################################################################
class ToyLoader(BaseLoader):
    def __init__(self,
                 root: str='./data/',
                 filename: str='toy',
                 test_size: float=0.2,
                 random_state: int=42):

        super().__init__()

        self.train_set = ToyDataset(root, filename, 1, test_size, random_state)
        self.test_set = ToyDataset(root, filename, 0, test_size, random_state)
        self.all_set = ToyDataset(root,  filename, 2, test_size, random_state)


    def loaders(self,
                batch_size: int=12,
                shuffle_train: bool=True,
                shuffle_test: bool=False,
                shuffle_all: bool=True,
                num_workers: int=0):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size,
                                  shuffle=shuffle_train, num_workers=num_workers,
                                  drop_last=False)

        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size,
                                 shuffle=shuffle_test, num_workers=num_workers,
                                 drop_last=False)

        all_loader = DataLoader(dataset=self.all_set, batch_size=batch_size,
                                shuffle=shuffle_all, num_workers=num_workers,
                                drop_last=False)

        return train_loader, test_loader, all_loader
