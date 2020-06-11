# 将 Unet 与 Resnet系列 和 EfficientNet 结合 在多项任务中取得最好成绩 (其他系列还在路上)
> 核心改动已经提交到 [fastai](https://github.com/fastai/fastai)


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