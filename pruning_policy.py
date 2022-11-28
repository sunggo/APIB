import torch
import torch.nn as nn
from torchvision import models
#import cv2
import sys
import numpy as np

def pruning_policy(model, layer_index, weights, filter_index, device):
    prev_op = None
    offset = -1
    op = list(model.modules())[layer_index]
    op.weight.data = torch.from_numpy(weights).to(device)
    while layer_index + offset >= 0:
        name,prev_op = list(model.named_modules())[layer_index + offset]
        # print(prev_op)
        if type(prev_op) == nn.Conv2d or type(prev_op) == nn.Linear:
            prev_op.weight.data = torch.from_numpy(prev_op.weight.data.cpu().numpy()[filter_index]).to(device)
            if prev_op.bias is not None:
                prev_op.bias.data = torch.from_numpy(prev_op.bias.data.cpu().numpy()[filter_index]).to(device)
            if name.endswith('downsample.conv'):
                offset-=1
                continue
            
            break
        # select bn
        elif type(prev_op) == nn.BatchNorm2d:
            prev_op.weight.data = torch.from_numpy(prev_op.weight.data.cpu().numpy()[filter_index]).to(device)
            prev_op.bias.data = torch.from_numpy(prev_op.bias.data.cpu().numpy()[filter_index]).to(device)
            prev_op.running_mean.data = torch.from_numpy(prev_op.running_mean.data.cpu().numpy()[filter_index]).to(device)
            prev_op.running_var.data = torch.from_numpy(prev_op.running_var.data.cpu().numpy()[filter_index]).to(device)
        offset -= 1
