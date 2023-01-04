# https://rwightman.github.io/pytorch-image-models/models/hrnet/

import torch
import timm

class W18_HRNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = timm.create_model('hrnet_w18', num_classes=11, head='segmentation', pretrained=True)

    def forward(self, x):
        x = self.segbackbone(x)    

        return {
            'out' : x
        }

