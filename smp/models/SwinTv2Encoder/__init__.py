from .encoder import *
import segmentation_models_pytorch as smp


# swin_transformer_v2_t
# swin_transformer_v2_s
# swin_transformer_v2_b
smp.encoders.encoders.update(SwinTv2_encoders)