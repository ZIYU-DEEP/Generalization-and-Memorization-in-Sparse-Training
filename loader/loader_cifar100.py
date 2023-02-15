"""
Title: CIFAR100_loader.py
Description: The loader classes for the CIFAR-10 datasets
Note: Haven't test the file yet! (Feb 13, 2022)
"""

from .loader_base import BaseLoader
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision.datasets import CIFAR100

import torchvision.transforms as transforms

# ----------- Handle the TPU Training Framework ----------- #
from torch.utils.data.distributed import DistributedSampler
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    SERIAL_EXEC = xmp.MpSerialExecutor()
except:
    pass
# --------------------------------------------------------- #


# #########################################################################
# 1. CIFAR100 Dataset
# #########################################################################
class CIFAR100Dataset(CIFAR100):
    """
    Add an index to get item.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if self.transform is not None: img = self.transform(img)
        return img, int(target), index


# #########################################################################
# 2. CIFAR100 Loader for Training
# #########################################################################
class CIFAR100Loader(BaseLoader):
    def __init__(self,
                 root: str='./data/',
                 filename: str='CIFAR100',
                 random_state: int=42):
        super().__init__()

        # Initialization
        self.root = root
        self.path = root + filename
        self.random_state = random_state
        self.n_classes = 100

        # Set the transform
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))])

        # This is simply to handle the TPU case
        def get_dataset():
            train_set = CIFAR100Dataset(root=self.root,
                                        train=True,
                                        transform=transform_train,
                                        download=True)

            test_set = CIFAR100Dataset(root=self.root,
                                       train=False,
                                       transform=transform_test,
                                       download=True)

            all_set = None  # Will remove this redundancy in future commits

            return train_set, test_set, all_set

        # Get the designated set
        try:
            self.train_set, self.test_set, self.all_set = SERIAL_EXEC.run(get_dataset)
        except:
            self.train_set, self.test_set, self.all_set = get_dataset()


    def loaders(self,
                batch_size: int=128,
                shuffle_train: bool=True,
                shuffle_test: bool=False,
                shuffle_all: bool=True,
                num_workers: int=0):
        """
        Construct loaders for training and testing.
        """

        # Set sampler for TPU
        try:
            sampler = DistributedSampler(self.train_set,
                                         num_replicas=xm.xrt_world_size(),
                                         rank=xm.get_ordinal(),
                                         shuffle=True)
            train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size,
                                      shuffle=shuffle_train, num_workers=num_workers,
                                      drop_last=False, pin_memory=True, sampler=sampler)
        except:
            train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size,
                                      shuffle=shuffle_train, num_workers=num_workers,
                                      drop_last=False, pin_memory=True)

        # Use default setting for test loader
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size,
                                 shuffle=shuffle_test, num_workers=num_workers,
                                 drop_last=False, pin_memory=True)

        all_loader = None  # Will remove this redundancy in future commits

        return train_loader, test_loader, all_loader


# #########################################################################
# 3. CIFAR100 Loader for Training with TPUs
# #########################################################################
class CIFAR100TPULoader(BaseLoader):
    def __init__(self,
                 root: str='./data/',
                 filename: str='CIFAR100',
                 random_state: int=42):
        super().__init__()

        # Initialization
        self.root = root
        self.path = root + filename
        self.random_state = random_state

        # Set the transform
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))])

        # Read in the data
        def get_dataset():
            train_set = CIFAR100Dataset(root=self.root,
                                        train=True,
                                        transform=transform_train,
                                        download=True)

            test_set = CIFAR100Dataset(root=self.root,
                                       train=False,
                                       transform=transform_test,
                                       download=True)

            all_set = None

            return train_set, test_set, all_set

        # Get the designated set
        try:
            self.train_set, self.test_set, self.all_set = SERIAL_EXEC.run(get_dataset)
        except:
            self.train_set, self.test_set, self.all_set = get_dataset()

    def loaders(self,
                batch_size: int=128,
                shuffle_train: bool=True,
                shuffle_test: bool=False,
                shuffle_all: bool=True,
                num_workers: int=0):
        # Set sampler
        try:
            train_sampler = DistributedSampler(self.train_set,
                                               num_replicas=xm.xrt_world_size(),
                                               rank=xm.get_ordinal(),
                                               shuffle=True)
            train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size,
                                      shuffle=shuffle_train, num_workers=num_workers,
                                      drop_last=False, pin_memory=True,
                                      sampler=train_sampler)
        except:
            train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size,
                                      shuffle=shuffle_train, num_workers=num_workers,
                                      drop_last=False, pin_memory=True)

        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size,
                                 shuffle=shuffle_test, num_workers=num_workers,
                                 drop_last=False, pin_memory=True)

        all_loader = None

        return train_loader, test_loader, all_loader
