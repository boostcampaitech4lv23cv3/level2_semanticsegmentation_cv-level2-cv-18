# https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py
import os
import torch
import numpy as np
import math
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from itertools import chain
from models.__uper_net__ import *
import segmentation_models_pytorch as smp

class Resnet101_UperNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = Resnet_UperNet_Base(num_classes=11, in_channels=3, backbone='resnet101')
        
    def forward(self, x):
        x = self.segbackbone(x)    

        return {
            'out' : x
        }
