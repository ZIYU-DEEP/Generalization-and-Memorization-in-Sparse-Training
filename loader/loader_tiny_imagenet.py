"""
Title: loader_tiny_imagent.py
Description: The loader classes for the imagenet datasets.
"""

from .loader_base import BaseLoader
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data.distributed import DistributedSampler

import os
import subprocess
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
# 1. TinyImageNet Dataset
# #########################################################################
class TinyImageNetDataset(ImageFolder):
    """
    Add an index to get item.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super(TinyImageNetDataset, self).__getitem__(index)
        return img, int(target), index


# #########################################################################
# 2. TinyImageNet Loader for Training
# #########################################################################
class TinyImageNetLoader(BaseLoader):
    def __init__(self,
                 root: str='/net/projects/yuxinlab/data/',
                 filename: str='tiny-imagenet-200',
                 random_state: int=42,
                 download: int=0):

        super().__init__()

        # Initialization
        self.root = root
        self.path = root + filename
        self.random_state = random_state

        # Dowload the dataset
        if download:
            print('Downloading Tiny imagenet dataset...')
            download_script = './scripts/data/tiny-imagenet.sh'
            os.chmod(download_script, 0o775)
            subprocess.check_call("%s %s %s" % (download_script, str(root), str(filename)),
                                  shell=True)
            print('Dataset downloaded and unzipped.')


        # Set the transform
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        # Read in the data
        def get_dataset():
            train_set = TinyImageNetDataset(Path(self.path) / 'train',
                                            transform_train)

            test_set = TinyImageNetDataset(Path(self.path) / 'val',
                                           transform_test)

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

        # Set sampler for the train loader
        try:
            train_sampler = DistributedSampler(self.train_set,
                                               num_replicas=xm.xrt_world_size(),
                                               rank=xm.get_ordinal(),
                                               shuffle=True)

        except:
            # Add distributed as an argument
            # train_sampler = DistributedSampler(self.train_set) if distributed else None
            train_sampler = None

        # Set the train loader
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size,
                                  shuffle=shuffle_train, num_workers=num_workers,
                                  drop_last=False, pin_memory=True,
                                  sampler=train_sampler)

        # Set the test loader
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size,
                                 shuffle=shuffle_test, num_workers=num_workers,
                                 drop_last=False, pin_memory=True)

        all_loader = None  # Deprecate this in future edits

        return train_loader, test_loader, all_loader
