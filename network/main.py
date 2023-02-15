"""
Title: main_network.py
Description: Build networks.
"""
from .mlp import MLP
from .alex_net import AlexNet
from .res_net import ResNet
from .preres_net import PreResNet
from .dense_net import DenseNet, BasicBlock
from .vgg_net import VGG, make_layers, cfg
from .mnist_lenet import MNISTLeNet
from .mnist_alexnet import MNISTAlexNet

def build_network(net_name='mlp',
                  in_dim: int=12,
                  out_dim: int=2,
                  hidden_act: str='tanh',
                  out_act: str='softmax',
                  hidden_dims: str='10-7-5-4-3',
                  depth: int=32,
                  widen_factor: int=4,
                  dropRate: int=0,
                  growthRate: int=12,
                  compressionRate: int=1):

    net_name = net_name.strip()

    if net_name == 'mlp':
        return MLP(in_dim,
                   out_dim,
                   hidden_act,
                   out_act,
                   hidden_dims)

    if net_name == 'alexnet':
        return AlexNet(out_dim)

    if net_name == 'preresnet':
        return PreResNet(depth,
                         out_dim)

    if net_name == 'resnet':
        return ResNet(depth,
                      out_dim,
                      widen_factor)

    if net_name == 'densenet':
        return DenseNet(depth,
                        BasicBlock,
                        dropRate,
                        out_dim,
                        growthRate,
                        compressionRate)

    if net_name == 'vgg11':
        return VGG(make_layers(cfg['A'], batch_norm=False),
                   out_dim)

    if net_name == 'vgg11_bn':
        return VGG(make_layers(cfg['A'], batch_norm=True),
                   out_dim)

    if net_name == 'vgg13':
        return VGG(make_layers(cfg['B'], batch_norm=False),
                   out_dim)

    if net_name == 'vgg13_bn':
        return VGG(make_layers(cfg['B'], batch_norm=True),
                   out_dim)

    if net_name == 'vgg16':
        return VGG(make_layers(cfg['D'], batch_norm=False),
                   out_dim)

    if net_name == 'vgg16_bn':
        return VGG(make_layers(cfg['D'], batch_norm=True),
                   out_dim)

    if net_name == 'vgg19':
        return VGG(make_layers(cfg['E'], batch_norm=False),
                   out_dim)

    if net_name == 'vgg19_bn':
        return VGG(make_layers(cfg['E'], batch_norm=True),
                   out_dim)

    if net_name == 'mnist_lenet':
        return MNISTLeNet()

    if net_name == 'mnist_alexnet':
        return MNISTAlexNet()

    return None
