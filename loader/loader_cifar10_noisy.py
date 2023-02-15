"""
Title: loader_cifar10_noisy.py
Description: The loader classes for the CIFAR-10 datasets.
"""

from .loader_base import BaseLoader
from PIL import Image
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import CIFAR10
import torchvision
import numpy as np
import torch
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
# 1. CIFAR10 Dataset
# #########################################################################
class CIFAR10NoisyDataset(CIFAR10):
    """
    Add an index to get item.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets_noisy = self.targets[:]

    def __getitem__(self, index):
        """
        We add `targets_noisy` and `is_noisy` compared to default.
        """

        # Get the image to be tensors
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        # Get the targets and indicators
        target = int(self.targets[index])
        targets_noisy = int(self.targets_noisy[index])
        is_noisy = int(target != targets_noisy)

        return img, target, index, targets_noisy, is_noisy


# #########################################################################
# 2. CIFAR10 Loader for Training
# #########################################################################
class CIFAR10NoisyLoader(BaseLoader):
    def __init__(self,
                 root: str='./data/',
                 filename: str='CIFAR10',
                 random_state: int=42,
                 noise_method: str='symmetric',
                 noise_ratio: float=0.2):
        """
        The dataset with the function of adding noise.
        """
        super().__init__()

        # Initialization
        self.n_classes = 10
        self.root = root
        self.path = root + filename
        self.random_state = random_state
        self.noise_method = noise_method
        self.noise_ratio = noise_ratio
        self.inds_noisy = []
        self.inds_clean = []

        # Set the transform
        transform = transforms.Compose([transforms.ToTensor()])

        # This is simply to handle the TPU case
        def get_dataset():
            train_set = CIFAR10NoisyDataset(root=self.root,
                                            train=True,
                                            transform=transform,
                                            download=True)

            test_set = CIFAR10NoisyDataset(root=self.root,
                                           train=False,
                                           transform=transform,
                                           download=True)

            all_set = None  # Will remove this redundancy in future commits

            return train_set, test_set, all_set

        # Get the designated set
        try:
            self.train_set, self.test_set, self.all_set = SERIAL_EXEC.run(get_dataset)
        except:
            self.train_set, self.test_set, self.all_set = get_dataset()

        # Add noise
        if self.noise_method == 'symmetric':
            self.add_symmetric_noise()

        elif self.noise_method == 'asymmetric':
            self.add_asymmetric_noise()

        # Create the clean and noisy loader (all from train set) separately
        self.clean_set = Subset(self.train_set, self.inds_clean)
        self.noisy_set = Subset(self.train_set, self.inds_noisy)


    def add_symmetric_noise(self):
        """
        Given a noise ratio, we randomly flip the original label
        of a data point to another label (can be the same).

        We only do this for the training set.
        This will also update self.inds_clean and self.inds_noisy.
        """

        # Get the placeholder for noisy targets
        targets_noisy = tuple(self.train_set.targets)[:]
        targets_noisy = list(targets_noisy)
        num_targets = len(targets_noisy)

        # Get the index to permute
        inds = np.random.permutation(len(targets_noisy))
        inds_noisy = inds[:int(len(inds) * self.noise_ratio)]

        # Random sample one label to replace the original
        for i in inds_noisy:
            targets_noisy[i] = np.random.choice(self.n_classes)

            # Record inds of the true noisy (label flipped ones)
            if targets_noisy[i] != self.train_set.targets[i]:
                self.inds_noisy.append(i)

        # Change the attribute values
        self.train_set.targets_noisy = targets_noisy

        # Record the inds of the clean data
        self.inds_clean = list(set(range(len(targets_noisy))) -
                               set(self.inds_noisy))

        return None


    def add_asymmetric_noise(self):
        """
        Given a noise ratio, for a certain class,
        we flip its label to another designated label.

        We only do this for the training set.
        This will also update self.inds_clean and self.inds_noisy.
        """

        # Get the placeholder for noisy targets
        targets_noisy = tuple(self.train_set.targets)[:]
        targets_noisy = list(targets_noisy)
        num_targets = len(targets_noisy)

        # Get the index to permute
        inds = np.random.permutation(len(targets_noisy))
        inds_noisy = inds[:int(len(inds) * self.noise_ratio)]

        # Change label to a designated one
        for i in inds_noisy:
            # Get the true label
            true_label = targets_noisy[i]

            # truck -> automobile
            if true_label == 9: targets_noisy[i] = 1

            # bird -> airplane
            elif true_label == 2: targets_noisy[i] = 0

            # cat -> dog
            elif true_label == 3: targets_noisy[i] = 5

            # dog -> cat
            elif true_label == 5: targets_noisy[i] = 3

            # deer -> horse
            elif true_label == 4: targets_noisy[i] = 7

            # Record inds of the true noisy (label flipped ones)
            if targets_noisy[i] != self.train_set.targets[i]:
                self.inds_noisy.append(i)

        # Change the attribute values
        self.train_set.targets_noisy = targets_noisy

        # Record the inds of the clean data
        self.inds_clean = list(set(range(len(targets_noisy))) -
                               set(self.inds_noisy))
        return None


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


    def loaders_noisy(self,
                      batch_size: int=128,
                      num_workers: int=0):
        """
        We additionally create two loaders solely from the training set.
        Clean loader contains the data with clean label.
        Noisy loader contains the data whose label != original label.
        """

        # Set sampler for TPU

        clean_loader = DataLoader(dataset=self.clean_set, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  drop_last=False, pin_memory=True)

        # Use default setting for test loader
        noisy_loader = DataLoader(dataset=self.noisy_set, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  drop_last=False, pin_memory=True)

        all_loader = None  # Will remove this redundancy in future commits

        return clean_loader, noisy_loader, all_loader
