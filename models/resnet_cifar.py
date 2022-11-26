import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import random

cifar_cfg = {
    20:[16,16,16,16,16,16,16,32,32,32,32,32,32,64,64,64,64,64,64],
    56:[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64],
    44:[16,16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    110:[16,16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
}

def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride=1, in_planes_new=None,affine=True,sample=False):
        super(BasicBlock, self).__init__()
        self.in_planes=in_planes
        self.sample_nums=0
        mid_planes, planes = out_planes[0], out_planes[1]
        conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(mid_planes, affine=affine)
        conv2 = nn.Conv2d(mid_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(planes, affine=affine)
        
        
        self.conv_bn1 = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1)]))
        self.conv_bn2 = nn.Sequential(OrderedDict([('conv', conv2), ('bn', bn2)]))
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != planes:
            if stride != 1:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, 0, 0, (planes - in_planes) // 2,
                                     planes - in_planes - (planes - in_planes) // 2), "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, 0, 0, (planes - in_planes) // 2,
                                     planes - in_planes - (planes - in_planes) // 2), "constant", 0))

    def forward(self, x):
        out = F.relu(self.conv_bn1(x))
        out = self.conv_bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_CIFAR(nn.Module):
    def __init__(self, depth=20, num_classes=10, cfg=None, in_cfg=None,cutout=False):
        super(ResNet_CIFAR, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth-2)//3
        self.depth=depth
        num_blocks = []
        if depth==20:
            num_blocks = [3, 3, 3]
        elif depth==32:
            num_blocks = [5, 5, 5]
        elif depth==44:
            num_blocks = [7, 7, 7]
        elif depth==56:
            num_blocks = [9, 9, 9]
        elif depth==110:
            num_blocks = [18, 18, 18]
        block = BasicBlock
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.cutout = cutout
        self.cfg = cfg
        self.in_cfg=in_cfg
        self.in_planes = 16
        conv1 = nn.Conv2d(3, cfg[0], kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(cfg[0])
        self.conv_bn = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1)]))
        if in_cfg == None:
            self.layer1 = self._make_layer(block, cfg[0:n], cfg[1:n+1], num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, cfg[n:2*n],cfg[n+1:2*n+1], num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, cfg[2*n:3*n], cfg[2*n+1:],num_blocks[2], stride=2)
        else:
            self.layer1 = self._make_layer_X(block, cfg[0:n],in_cfg[0:int(n/2)], cfg[1:n+1], num_blocks[0], stride=1)
            self.layer2 = self._make_layer_X(block, cfg[n:2*n],in_cfg[int(n/2):n],cfg[n+1:2*n+1], num_blocks[1], stride=2)
            self.layer3 = self._make_layer_X(block, cfg[2*n:3*n],in_cfg[n:int(3*n/2)], cfg[2*n+1:],num_blocks[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(cfg[-1], num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, in_planes, out_planes,num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        count = 0
        for i, stride in enumerate(strides):
            layers.append(('block_%d'%i,block(in_planes[count], out_planes[count:count+2], stride)))
            count += 2
        return nn.Sequential(OrderedDict(layers))
    def _make_layer_X(self, block, in_planes_o,in_planes_p, out_planes,num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        count = 0
        in_count=0
        for i, stride in enumerate(strides):
            layers.append(('block_%d'%i,block(in_planes_o[count], out_planes[count:count+2], stride,in_planes_new=in_planes_p[in_count],sample=True)))
            count += 2
            in_count+=1
        return nn.Sequential(OrderedDict(layers))
    def forward(self, x):
        
        out = F.relu(self.conv_bn(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    

    @property
    def config(self):
        return {
            'name': self.__class__.__name__,
            'cfg': self.cfg,
            'cfg_base': self.cfg_base,
            'dataset': 'cifar10',
        }
def resnet20():
    return ResNet_CIFAR(depth=20,cfg=cifar_cfg[20])

def resnet20_X(nums=None):
    return ResNet_CIFAR(depth=20,cfg=nums)
def resnet56():
    return ResNet_CIFAR(depth=56,cfg=cifar_cfg[56])
def resnet56_X(nums=None):
    return ResNet_CIFAR(depth=56,cfg=nums)
def resnet44():
    return ResNet_CIFAR(depth=44,cfg=cifar_cfg[44])
def resnet44_X(nums=None):
    return ResNet_CIFAR(depth=44,cfg=nums)
def resnet110():
    return ResNet_CIFAR(depth=110,cfg=cifar_cfg[110])
def resnet110_X(nums=None):
    return ResNet_CIFAR(depth=110,cfg=nums)