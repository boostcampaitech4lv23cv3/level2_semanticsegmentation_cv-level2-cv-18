import torch
from torchvision import models
import segmentation_models_pytorch as smp

class LRASPP_Mobilenetv3(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = models.segmentation.lraspp_mobilenet_v3_large(num_classes=11)
        
    def forward(self, x):
        x = self.segbackbone(x)    
        return x