import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import sys
import numpy as np
import torch
def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes_1, planes_2=0, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        conv1 = conv3x3(inplanes, planes_1, stride)
        bn1 = norm_layer(planes_1)
        relu = nn.ReLU()
        if planes_2 == 0:
            conv2 = conv3x3(planes_1, inplanes)
            bn2 = norm_layer(inplanes)
        else:
            conv2 = conv3x3(planes_1, planes_2)
            bn2 = norm_layer(planes_2)
        self.relu = relu
        self.conv1 = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1), ('relu', relu)]))
        self.conv2 = nn.Sequential(OrderedDict([('conv', conv2), ('bn', bn2)]))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes_1, planes_2, planes_3=0, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        conv1 = conv1x1(inplanes, planes_1)
        bn1 = norm_layer(planes_1)
        conv2 = conv3x3(planes_1, planes_2, stride)
        bn2 = norm_layer(planes_2)
        if planes_3 == 0:
            conv3 = conv1x1(planes_2, inplanes)
            bn3 = norm_layer(inplanes)
        else:
            conv3 = conv1x1(planes_2, planes_3)
            bn3 = norm_layer(planes_3)
        relu = nn.ReLU()
        self.relu = relu
        self.conv1 = nn.Sequential(OrderedDict([('conv', conv1), ('bn', bn1), ('relu', relu)]))
        self.conv2 = nn.Sequential(OrderedDict([('conv', conv2), ('bn', bn2), ('relu', relu)]))
        self.conv3 = nn.Sequential(OrderedDict([('conv', conv3), ('bn', bn3)]))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet_ImageNet(nn.Module):
    def __init__(self, cfg=None, depth=18, block=BasicBlock, num_classes=1000):
        super(ResNet_ImageNet, self).__init__()
        self.cfgs_base = {18: [64, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512],
                          34: [64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512],
                          50: [64, 64, 64, 256, 64, 64, 64, 64, 128, 128, 512, 128, 128, 128, 128, 128, 128, 256, 256, 1024, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 2048, 512, 512, 512, 512]}
        if depth==18:
            block = BasicBlock
            blocks = [2, 2, 2, 2]
            _cfg = self.cfgs_base[18]
        elif depth==34:
            block = BasicBlock
            blocks = [3, 4, 6, 3]
            _cfg = self.cfgs_base[34]
        elif depth==50:
            block = Bottleneck
            blocks = [3, 4, 6, 3]
            _cfg = self.cfgs_base[50]
        if cfg == None:
            cfg = _cfg
        norm_layer = nn.BatchNorm2d
        self.num_classes = num_classes
        self._norm_layer = norm_layer
        self.depth = depth
        self.cfg = cfg
        self.inplanes = cfg[0]
        self.blocks = blocks
        self.conv1 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                                                ('bn', norm_layer(self.inplanes)),
                                                ('relu', nn.ReLU())]))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if depth!=50:
            self.layer1 = self._make_layer(block, cfg[1 : blocks[0]+2], blocks[0])
            self.layer2 = self._make_layer(block, cfg[blocks[0]+2 : blocks[0]+2+blocks[1]+1], blocks[1], stride=2,)
            self.layer3 = self._make_layer(block, cfg[blocks[0]+blocks[1]+3 : blocks[0]+blocks[1]+blocks[2]+4], blocks[2], stride=2,)
            self.layer4 = self._make_layer(block, cfg[blocks[0]+blocks[1]+blocks[2]+4: ], blocks[3], stride=2,)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(cfg[blocks[0]+blocks[1]+blocks[2]+5], num_classes)
        else:
            self.layer1 = self._make_layer(block, cfg[1 : 2*blocks[0]+2], blocks[0])
            self.layer2 = self._make_layer(block, cfg[2*blocks[0]+2 : 2*blocks[0]+2+2*blocks[1]+1], blocks[1], stride=2,)
            self.layer3 = self._make_layer(block, cfg[2*blocks[0]+2*blocks[1]+3 : 2*blocks[0]+2*blocks[1]+2*blocks[2]+4], blocks[2], stride=2,)
            self.layer4 = self._make_layer(block, cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+4: ], blocks[3], stride=2,)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(cfg[2*blocks[0]+2*blocks[1]+2*blocks[2]+6], num_classes)
        

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if self.depth == 50:
            first_planes = planes[0:3]
            # downsample at each 1'st layer, for pruning
            downsample = nn.Sequential(OrderedDict([('conv', conv1x1(self.inplanes, first_planes[-1], stride)),
                                                    ('bn', norm_layer(first_planes[-1]))]))
            layers = []
            layers.append(block(self.inplanes, first_planes[0], first_planes[1], first_planes[2], stride, downsample, norm_layer))
            self.inplanes = first_planes[-1]
            later_planes = planes[3:3+2*(blocks-1)]
            for i in range(1, blocks):
                layers.append(block(self.inplanes, later_planes[2*(i-1)], later_planes[2*(i-1)+1], norm_layer=norm_layer))
            return nn.Sequential(*layers)
        else:
            first_planes = planes[0:2]
            # downsample at each 1'st layer, for pruning
            downsample = nn.Sequential(OrderedDict([('conv', conv1x1(self.inplanes, first_planes[-1], stride)),
                                                    ('bn', norm_layer(first_planes[-1]))]))
            layers = []
            layers.append(block(self.inplanes, first_planes[0], first_planes[1], stride, downsample, norm_layer))
            self.inplanes = first_planes[-1]
            later_planes = planes[2:2+blocks-1]
            for i in range(1, blocks):
                layers.append(block(self.inplanes, later_planes[i-1], norm_layer=norm_layer))
            return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    @property
    def config(self):
        return {
            'name': self.__class__.__name__,
            'depth': self.depth,
            'cfg': self.cfg,
            'cfg_base': self.cfgs_base[self.depth],
            'dataset': 'ImageNet',
        }
def resnet50():
    return ResNet_ImageNet(depth=50)
def resnet50_X(cfg):
    return ResNet_ImageNet(depth=50,cfg=cfg)