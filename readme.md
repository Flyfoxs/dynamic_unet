# 将 Unet与预训练网络结合, 在多项器官分割中取得优异成绩
-  测试过的预训练网络包括(如有其他需求可提交Issue) 
    - VGG
    - Resnet
    - Densenet
    - Efficientnet
    
-  核心改动已经提交到 [fastai](https://github.com/fastai/fastai)

# Resnet
```python
encoder = efficient_unet(0)
unet = DynamicUnet(encoder, n_classes=5, img_size=(224, 224), blur=False, blur_final=False,
                self_attention=False, y_range=None, norm_type=NormType,
                last_cross=True,
                bottle=False)

print(unet(torch.rand(1,3,224,224)).shape)
```



# EfficientNet
```python
encoder = nn.Sequential(*list(models.resnet34().children())[:-3])

unet = DynamicUnet(encoder, n_classes=5, img_size=(224, 224), blur=False, blur_final=False,
                    self_attention=False, y_range=None, norm_type=NormType,
                    last_cross=True,
                    bottle=False)
print(unet(torch.rand(1,3,224,224)).shape)
```

更多其他网络参考: [notebook](https://github.com/Flyfoxs/dynamic_unet/blob/master/notebook/different_network.ipynb)  
