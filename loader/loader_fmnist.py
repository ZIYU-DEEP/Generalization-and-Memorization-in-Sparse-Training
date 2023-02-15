"""
Title: fMNIST_loader.py
Description: The loader classes for the MNIST datasets.
Note: Haven't test the file yet! (Feb 13, 2022)
"""

from .loader_base import BaseLoader
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision.datasets import FashionMNIST

import torchvision.transforms as transforms


# #########################################################################
# 1. MNIST Dataset
# #########################################################################
class FashionMNISTDataset(FashionMNIST):
    """
    Add an index to get item.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None: img = self.transform(img)
        return img, int(target), index


# #########################################################################
# 2. MNIST Loader for Training
# #########################################################################
class FashionMNISTLoader(BaseLoader):
    def __init__(self,
                 root: str='./data/',
                 filename: str='FasionMNIST',
                 random_state: int=42):
        super().__init__()

        # Initialization
        self.root = root
        self.path = root + filename
        self.random_state = random_state
        self.n_classes = 10

        # Set the transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        # Read in the data
        train_set = FashionMNISTDataset(root=self.root,
                                        train=True,
                                        transform=transform,
                                        download=True)

        test_set = FashionMNISTDataset(root=self.root,
                                       train=False,
                                       transform=transform,
                                       download=True)

        all_set = ConcatDataset([train_set, train_set])

        # Get the designated set
        self.train_set = train_set
        self.test_set = test_set
        self.all_set = all_set

    def loaders(self,
                batch_size: int=128,
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
