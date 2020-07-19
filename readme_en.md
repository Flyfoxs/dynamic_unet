[Chinese version](https://github.com/Flyfoxs/dynamic_unet/blob/master/readme.md)
# Take pretrained model as encoder of UNET, get better score in many organ segmentation task
-  Already test on these models 
    - VGG
    - Resnet
    - Densenet
    - Efficientnet
    
    
# Install
```shell script
pip install git+https://github.com/Flyfoxs/dynamic_unet@master
```

-  The core code submit to fastai [fastai](https://github.com/fastai/fastai)

# EfficientNet
```python
encoder = efficient_unet(0)
unet = DynamicUnet(encoder, n_classes=5, img_size=(224, 224), blur=False, blur_final=False,
                self_attention=False, y_range=None, norm_type=NormType,
                last_cross=True,
                bottle=False)

print(unet(torch.rand(1,3,224,224)).shape)
```

# Densenet
```python
encoder = nn.Sequential(*list(models.densenet121().children())[0])
unet = DynamicUnet(encoder, n_classes=5, img_size=(224, 224), blur=False, blur_final=False,
                    self_attention=False, y_range=None, norm_type=NormType,
                    last_cross=True,
                    bottle=False)
print(unet(torch.rand(1,3,224,224)).shape)
```

# Resnet
```python
encoder = nn.Sequential(*list(models.resnet34().children())[:-3])

unet = DynamicUnet(encoder, n_classes=5, img_size=(224, 224), blur=False, blur_final=False,
                    self_attention=False, y_range=None, norm_type=NormType,
                    last_cross=True,
                    bottle=False)
print(unet(torch.rand(1,3,224,224)).shape)
```

You can get more network example: [notebook](https://github.com/Flyfoxs/dynamic_unet/blob/master/notebook/different_network.ipynb)  
