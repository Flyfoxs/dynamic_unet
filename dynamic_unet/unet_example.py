from .base import *
import torch
from torch import nn
from fastai.vision import models
from efficientnet_pytorch import EfficientNet

# Take EfficientNet as encode
for i in range(3):
    encode = efficient_unet(0)
    unet = DynamicUnet(encoder, n_classes=5, img_size=(224, 224), blur=False, blur_final=False,
                    self_attention=False, y_range=None, norm_type=NormType,
                    last_cross=True,
                    bottle=False)

    print(unet(torch.rand(1,3,224,224)).shape)