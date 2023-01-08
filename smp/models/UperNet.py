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
from .TimmEncoder import *

class Resnet101_UperNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = Resnet_UperNet_Base(num_classes=11, in_channels=3, backbone='resnet101')
        
    def forward(self, x):
        x = self.segbackbone(x)    

        return {
            'out' : x
        }


class TimmSwinTv2w24i384_UperNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = TimmSwinTv2_Encoder(model = "swinv2_base_window12to24_192to384_22kft1k")
        self.feature_channels = self.encoder._out_channels
        self.model = Custom_UperNet(backbone = self.encoder, feature_channels = self.feature_channels, fpn_out = self.feature_channels[0])
        
    def forward(self, x):
        x = self.model(x)    

        return {
            'out' : x
        }
