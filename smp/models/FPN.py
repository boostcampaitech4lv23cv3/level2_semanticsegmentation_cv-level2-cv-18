import torch
import segmentation_models_pytorch as smp

class Efficientb2_FPN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = smp.FPN(
                                    encoder_name="efficientnet-b2", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                                    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                    classes=11,                     # model output channels (number of classes in your dataset)
                                )
        
    def forward(self, x):
        x = self.segbackbone(x)    

        return {
            'out' : x
        }

class Efficientb3_FPN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = smp.FPN(
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

class Efficientb4_FPN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = smp.FPN(
                                    encoder_name="efficientnet-b4", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                                    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                    classes=11,                     # model output channels (number of classes in your dataset)
                                )
        
    def forward(self, x):
        x = self.segbackbone(x)    

        return {
            'out' : x
        }

class Resnet101_FPN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = smp.FPN(
                                    encoder_name="resnet101", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                                    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                    classes=11,                     # model output channels (number of classes in your dataset)
                                )
        
    def forward(self, x):
        x = self.segbackbone(x)    

        return {
            'out' : x
        }

class Resnext101_FPN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = smp.FPN(
                                    encoder_name="resnext101_32x8d", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                                    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                    classes=11,                     # model output channels (number of classes in your dataset)
                                )
        
    def forward(self, x):
        x = self.segbackbone(x)    

        return {
            'out' : x
        }



class TimmEfficientb3_FPN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = smp.FPN(
                                    encoder_name="timm-efficientnet-b3", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                                    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                    classes=11,                     # model output channels (number of classes in your dataset)
                                )
        
    def forward(self, x):
        x = self.segbackbone(x)    

        return {
            'out' : x
        }

class TimmEfficientb4_FPN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = smp.FPN(
                                    encoder_name="timm-efficientnet-b4", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                                    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                    classes=11,                     # model output channels (number of classes in your dataset)
                                )
        
    def forward(self, x):
        x = self.segbackbone(x)    

        return {
            'out' : x
        }

class Resnext50_FPN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = smp.FPN(
                                    encoder_name="resnext50_32x4d", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                                    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                    classes=11,                     # model output channels (number of classes in your dataset)
                                )
        
    def forward(self, x):
        x = self.segbackbone(x)    

        return {
            'out' : x
        }

class GERNet_FPN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = smp.FPN(
                                    encoder_name="timm-gernet_m", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                                    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                    classes=11,                     # model output channels (number of classes in your dataset)
                                )
        
    def forward(self, x):
        x = self.segbackbone(x)    

        return {
            'out' : x
        }


class SwinTv2t_FPN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = smp.FPN(
                                    encoder_name="swin_transformer_v2_t", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                                    encoder_weights = None,
                                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                    classes=11,                     # model output channels (number of classes in your dataset)
                                )
        
    def forward(self, x):
        x = self.segbackbone(x)    

        return {
            'out' : x
        }

class SwinTv2s_FPN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = smp.FPN(
                                    encoder_name="swin_transformer_v2_s", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                                    encoder_weights = None,
                                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                    classes=11,                     # model output channels (number of classes in your dataset)
                                )
        
    def forward(self, x):
        x = self.segbackbone(x)    

        return {
            'out' : x
        }

class TimmSwinTv2wbi256_FPN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = smp.FPN(
                                    encoder_name="timm_swinv2_w8i256", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                                    encoder_weights = None,
                                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                    classes=11,                     # model output channels (number of classes in your dataset)
                                )
        
    def forward(self, x):
        x = self.segbackbone(x)    

        return {
            'out' : x
        }

class TimmSwinTv2crs224_FPN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segbackbone = smp.FPN(
                                    encoder_name="timm_swinv2_cr_small_224", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                                    encoder_weights = None,
                                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                    classes=11,                     # model output channels (number of classes in your dataset)
                                )
        
    def forward(self, x):
        x = self.segbackbone(x)    

        return {
            'out' : x
        }
