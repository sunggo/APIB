# https://github.com/pytorch/vision/blob/master/torchvision/models
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo

# from utils import cutout_batch
import numpy as np

cifar_cfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}


class VGG_CIFAR(nn.Module):
    def __init__(self, cfg=None, cutout=True, num_classes=10):
        super(VGG_CIFAR, self).__init__()
        if cfg is None:
            cfg = [64, 64, 128, 128, 256, 256,
                   256, 512, 512, 512, 512, 512, 512]
        self.cutout = cutout
        self.cfg = cfg
        _cfg = list(cfg)
        self._cfg = _cfg
        self.feature = self.make_layers(_cfg, True)
        self.avgpool = nn.AvgPool2d(2)
        self.classifier = nn.Sequential(
            nn.Linear(self.cfg[-1], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        self.num_classes = num_classes
        self.classifier_param = (
            self.cfg[-1] + 1) * 512 + (512 + 1) * num_classes

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        pool_index = 0
        conv_index = 0
        for v in cfg:
            if v == 'M':
                layers += [('maxpool_%d' % pool_index,
                            nn.MaxPool2d(kernel_size=2, stride=2))]
                pool_index += 1
            else:
                conv2d = nn.Conv2d(
                    in_channels, v, kernel_size=3, padding=1, bias=False)
                conv_index += 1
                if batch_norm:
                    bn = nn.BatchNorm2d(v)
                    layers += [('conv_%d' % conv_index, conv2d), ('bn_%d' % conv_index, bn),
                               ('relu_%d' % conv_index, nn.ReLU(inplace=True))]
                else:
                    layers += [('conv_%d' % conv_index, conv2d),
                               ('relu_%d' % conv_index, nn.ReLU(inplace=True))]
                in_channels = v
        self.conv_num = conv_index
        return nn.Sequential(OrderedDict(layers))


    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y
    @property
    def config(self):
        return {
            'name': self.__class__.__name__,
            'cfg': self.cfg,
            'cfg_base': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
            'dataset': 'cifar10',
        }


def _vgg(arch, cfg):
    model = VGG_CIFAR(cfg)
    return model
def vgg16_X(nums=None):
    cfg=nums
    cfg.insert(2, 'M')
    cfg.insert(5, 'M')
    cfg.insert(9, 'M')
    cfg.insert(13, 'M')
    return _vgg('vgg16_X',cfg)
    
def vgg16():
    return _vgg('vgg16',cifar_cfg[16])