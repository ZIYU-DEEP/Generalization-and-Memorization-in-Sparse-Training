# Generalization and Memorization in Sparse Neural Networks
[![Python 3.8](https://img.shields.io/badge/Python-3.8-blueviolet.svg)](https://www.python.org/downloads/release/python-380/) [![Pytorch](https://img.shields.io/badge/Pytorch-1.12.1-critical.svg)](https://github.com/pytorch/pytorch/releases/tag/v1.12.0) [![License](https://img.shields.io/badge/License-Apache%202.0-ff69b4.svg)](https://opensource.org/licenses/Apache-2.0) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-success.svg)](https://github.com/ZIYU-DEEP/Generalization-and-Memorization-in-Sparse-Training)

![illustration](illustration.png)

This is the repository for our [paper](https://github.com/ZIYU-DEEP/Generalization-and-Memorization-in-Sparse-Training/blob/main/paper.pdf) ([poster](https://github.com/ZIYU-DEEP/Generalization-and-Memorization-in-Sparse-Training/blob/main/Poster.pdf)) on "***The Price of Sparsity: Generalization and Memorization in Sparse Neural Networks***", presented at the [Sparsity in Neural Networks Workshop](https://www.sparseneural.net/) (virtual + ICML meetup, July 13th 2022).

We will archive our paper and poster here, and release the code (in PyTorch and Jax) upon the finalization of the research project. In the meantime, if you would like to request any code or instruction to reimplement our experiments, please do not hesitate to contact me at ziyuye@uchicago.edu or ziyuye@live.com.

Below is a temporary README, and we will update it soon after the submission of the full paper.


## 1. Intro
This repository is organized as follows:
```bash
root
|== loader
│   └── loader_cifar100.py
│   └── loader_cifar100_noisy.py
│   └── main.py
|== network
│   └── mlp.py
│   └── resnet.py
│   └── main.py
|== optim
│   └── trainer.py
│   └── model.py
|== helper
│   └── pruner.py
│   └── utils.py
|== scripts
│   └── cifar100_resnet.sh
|== run.py
|== experiment.py
```


## 2. Requirements


### Working with CPU/GPU
If you are using anaconda:
```bash
conda create --name sparse python=3.8
conda activate sparse
```

To install necessary pakacges, check the list in `./requirements.txt` or lazily run the following in the designated environment for the project:
```bash
python3 -m pip install -r requirements.txt
```
If you do not want to run the whole `requirements.txt`, at least make sure you have the following uncommon pacakges installed:

```bash
python3 -m pip install joblib  # We will remove this package later to torch save
python3 -m pip install psutil
python3 -m pip install pyhessian
python3 -m pip install functorch  # Never run this on TPU machines!
python3 -m pip install git+https://github.com/tfjgeorge/nngeometry.git
python3 -m pip install git+https://github.com/noahgolmant/pytorch-hessian-eigenthings.git@master#egg=hessian-eigenthings
python3 -m pip install git+https://github.com/facebookresearch/jacobian_regularizer
```
### Working with TPU
[July 2022] If you are using TPUs on Google Cloud platform, please make sure you have also run the following (more information can be found [here](https://cloud.google.com/tpu/docs/run-calculation-pytorch#tpu-vm)).
```bash
# Config the TPU
echo "export XRT_TPU_CONFIG='localservice;0;localhost:51011'" >> ~/.bashrc
source ~/.bashrc

# Install torch_xla; you may install a previous version if the following does not work
pip install https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl

# Install cloud client for tpu
pip install cloud-tpu-client
```
When running experiments with TPUs, you should set `--device tpu` in the arguments of `run.py`. The current code is only tested on TPUs for training, thus you may better set `--save_snr 0` and `--save_fisher 0`, and let CPUs/GPUs do the work of calculating SNR and Fisher information.

### Troubleshooting for GPU NVIDIA A30/40/100
[September 2022] If you encounter the CUDA capability (usually happen for Nvidia A30/40 cards or so) issue with Torch, an easy fix may be:
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
However, this solution may be incompatible when using the `functorch==0.2.1` package which requires a torch version in between 1.12.1 and 1.13 (e.g., `torch==1.12.1+cu102`). A better solution should be to directly update the CUDA version.
