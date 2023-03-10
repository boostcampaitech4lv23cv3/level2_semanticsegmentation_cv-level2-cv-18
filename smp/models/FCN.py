import torch
from torchvision import models

class FCN_Resnet50(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = models.segmentation.fcn_resnet50(weights = models.segmentation.FCN_ResNet50_Weights.DEFAULT)
        # output class를 data set에 맞도록 수정
        self.segbackbone.classifier[4] = torch.nn.Conv2d(512, 11, kernel_size=1) # type: ignore

        
    def forward(self, x):
        x = self.segbackbone(x)    
        return x
        
class FCN_Resnet101(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = models.segmentation.fcn_resnet101(weights = models.segmentation.FCN_ResNet101_Weights.DEFAULT)
        # output class를 data set에 맞도록 수정
        self.segbackbone.classifier[4] = torch.nn.Conv2d(512, 11, kernel_size=1) # type: ignore

        
    def forward(self, x):
        x = self.segbackbone(x)    
        return x