import torch
import segmentation_models_pytorch as smp

class Efficientb3_Linknet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = smp.Linknet(
                                    encoder_name="efficientnet-b3", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                                    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                    classes=11,                     # model output channels (number of classes in your dataset)
                                )
        
    def forward(self, x):
        x = self.segbackbone(x)    

        return {
            'out' : x
        }

