from typing import Tuple, List

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders._base import EncoderMixin
from .model import *

class SwinTv2_Encoder(torch.nn.Module, EncoderMixin):

    def __init__(self, **kwargs):
        super().__init__()
        # Define encoder modules below
        self.model = SwinTransformerV2(input_resolution=kwargs["input_resolution"],
                             window_size=kwargs["window_size"],
                             in_channels=kwargs["in_channels"],
                             use_checkpoint=kwargs["use_checkpoint"],
                             sequential_self_attention=kwargs["sequential_self_attention"],
                             embedding_channels=kwargs["embedding_channels"],
                             depths=kwargs["depths"],
                             number_of_heads=kwargs["number_of_heads"])

        # A number of stages in decoder (in other words number of downsampling operations), integer
        # use in in forward pass to reduce number of returning features
        self._depth: int = len(self.model.stages) #- 1

        # Default number of input channels in first Conv2d layer for encoder (usually 3)
        self._in_channels: int = 3

        # A number of channels for each encoder feature tensor, list of integers
        
        self.emc:int = self.model.patch_embedding.out_channels # type: ignore
        self._out_channels: List[int] = [self.emc * (2 ** max(i, 0)) for i in range(self._depth)]


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = self.model(x)
        return list(outs)

SwinTv2_encoders = {
    "swin_transformer_v2_t": {
        "encoder": SwinTv2_Encoder,
        "pretrained_settings": {},
        "params": {
            "input_resolution": (512,512),
            "window_size": 8,
            "in_channels": 3,
            "use_checkpoint": False,
            "sequential_self_attention": False,
            "embedding_channels": 96, 
            "depths": (2, 2, 6, 2),
            "number_of_heads": (3, 6, 12, 24),
        },
    },
    "swin_transformer_v2_s": {
        "encoder": SwinTv2_Encoder,
        "pretrained_settings": {},
        "params": {
            "input_resolution": (512,512),
            "window_size": 8,
            "in_channels": 3,
            "use_checkpoint": False,
            "sequential_self_attention": False,
            "embedding_channels": 96,
            "depths": (2, 2, 18, 2),
            "number_of_heads": (3, 6, 12, 24),
        },
    },
    "swin_transformer_v2_b": {
        "encoder": SwinTv2_Encoder,
        "pretrained_settings": {},
        "params": {
            "input_resolution": (512,512),
            "window_size": 8,
            "in_channels": 3,
            "use_checkpoint": False,
            "sequential_self_attention": False,
            "embedding_channels": 128,
            "depths": (2, 2, 18, 2),
            "number_of_heads": (4, 8, 16, 32),
        },
    },
}