from typing import Tuple, List

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders._base import EncoderMixin
from . import timm
import torch.utils.checkpoint as checkpoint
from .timm.models.swin_transformer_v2 import SwinTransformerV2, BasicLayer
from .timm.models.swin_transformer_v2_cr import SwinTransformerV2Cr

class SwinTv2Stage(nn.Module):
    def __init__(self, downsample, blocks):
        super().__init__()
        self.downsample = downsample
        self.blocks = blocks
        self.grad_checkpointing = False

    def forward(self, x):
        x = self.downsample(x)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class TimmSwinTv2_Encoder(torch.nn.Module, EncoderMixin):

    def __init__(self, **kwargs):
        super().__init__()
        model = timm.create_model(kwargs["model"], pretrained=True)
        # Define encoder modules below
        print("model.layers", len(model.layers))
        print("model.patch_embed", model.patch_embed)
        print("model.embed_dim", model.embed_dim)
        self.model:SwinTransformerV2 = model
        self.patch_embed = model.patch_embed
        self.absolute_pos_embed = model.absolute_pos_embed if model.absolute_pos_embed is not None else None
        self.pos_drop = model.pos_drop
        self.layers = model.layers
        # A number of stages in decoder (in other words number of downsampling operations), integer
        # use in in forward pass to reduce number of returning features
        self._depth: int = self.model.num_layers #- 1

        # Default number of input channels in first Conv2d layer for encoder (usually 3)
        self._in_channels: int = 3

        # A number of channels for each encoder feature tensor, list of integers
        
        self.emc:int = self.model.embed_dim
        self._out_channels: List[int] = [self.emc * (2 ** max(i, 0)) for i in range(self._depth)]
        print("self._out_channels", self._out_channels)

        # swap patch_merge
        blocks_list = []
        downsample_list = []
        for layer in self.layers:
            downsample_list.append(layer.downsample)
            blocks_list.append(layer.blocks)
        downsample_list = [downsample_list[-1], *downsample_list[0:-1]]
        stages = []
        for blocks, downsample in zip(blocks_list, downsample_list):
            layer = SwinTv2Stage(downsample=downsample, blocks=blocks)
            stages.append(layer)
        self.stages = stages

# torch.Size([8, 4096, 128]) bad / batch, -1, emb
# torch.Size([8, 1024, 256]) 
# torch.Size([8, 256, 512])
# torch.Size([8, 64, 1024])
# 
# torch.Size([8, 96, 56, 56]) good / batch, emb, h,w
# torch.Size([8, 192, 28, 28])
# torch.Size([8, 384, 14, 14])
# torch.Size([8, 768, 7, 7])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.stages:
            x = layer(x)
            feature = x.transpose(1,2) # change h*w , emb -> emb , h*w
            shape = feature.shape
            hw = int(pow(shape[-1], 0.5))
            feature = feature.view(shape[0],shape[1], hw ,hw)
            outs.append(feature)
        return outs

class TimmSwinTv2Cr_Encoder(torch.nn.Module, EncoderMixin):

    def __init__(self, **kwargs):
        super().__init__()
        model:SwinTransformerV2Cr = timm.create_model(kwargs["model"], pretrained=True)
        # Define encoder modules below
        self.model:SwinTransformerV2Cr = model
        # self.model.update_input_size(new_img_size = (256,256))
        print("model.stages", len(self.model.stages))
        print("model.patch_embed", self.model.patch_embed)
        print("model.embed_dim", self.model.embed_dim)
        self.patch_embed = model.patch_embed
        self.layers = model.stages
        # A number of stages in decoder (in other words number of downsampling operations), integer
        # use in in forward pass to reduce number of returning features
        self._depth: int = len(self.model.stages) #- 1

        # Default number of input channels in first Conv2d layer for encoder (usually 3)
        self._in_channels: int = 3

        # A number of channels for each encoder feature tensor, list of integers
        
        self.emc:int = self.model.embed_dim
        self._out_channels: List[int] = [self.emc * (2 ** max(i, 0)) for i in range(self._depth)]
        print("self._out_channels", self._out_channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        return outs

# swinv2_base_window8_256
# swinv2_base_window12_192_22k
# swinv2_base_window12to16_192to256_22kft1k
# swinv2_base_window12to24_192to384_22kft1k
# swinv2_base_window16_256
# swinv2_cr_small_224
# swinv2_cr_small_ns_224
# swinv2_cr_tiny_ns_224
# swinv2_large_window12_192_22k
# swinv2_large_window12to16_192to256_22kft1k
# swinv2_large_window12to24_192to384_22kft1k
# swinv2_small_window8_256
# swinv2_small_window16_256
# swinv2_tiny_window8_256
# swinv2_tiny_window16_256

TimmSwinTv2_encoders = {
    "timm_swinv2_w8i256": {
        "encoder": TimmSwinTv2_Encoder,
        "pretrained_settings": {},
        "params": {
            "model": "swinv2_base_window8_256"
        },
    },
    "timm_swinv2_cr_small_224": {
        "encoder": TimmSwinTv2Cr_Encoder,
        "pretrained_settings": {},
        "params": {
            "model": "swinv2_cr_small_224"
        },
    },
}