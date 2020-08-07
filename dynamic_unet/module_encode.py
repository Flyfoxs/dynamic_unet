# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from fastai.vision import *
from fastai.vision.models import WideResNet
from fvcore.common.registry import Registry

UNET_ENCODE = Registry("UNET_ENCODE")

@UNET_ENCODE.register()
def resnet18():
    return nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])

@UNET_ENCODE.register()
def densenet121():
    return nn.Sequential(*list(models.densenet121(pretrained=True).children())[0])

@UNET_ENCODE.register()
def densenet169():
    return nn.Sequential(*list(models.densenet169(pretrained=True).children())[0])

@UNET_ENCODE.register()
def densenet201():
    return nn.Sequential(*list(models.densenet201(pretrained=True).children())[0])

@UNET_ENCODE.register()
def efficientnet(name='5'):
    from efficientnet_pytorch import EfficientNet
    class EfficientNet_(EfficientNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, inputs):
            x = self.extract_features(inputs)
            return x
    print('in_channels=', in_channels)
    return EfficientNet_.from_pretrained(f'efficientnet-b{name}')

@UNET_ENCODE.register()
def wrn_22():
    def _wrn_22():
        "Wide ResNet with 22 layers."
        return WideResNet(num_groups=3, N=3, num_classes=10, k=6, drop_p=0.2)

    return nn.Sequential(*list(_wrn_22().children())[0])


for i in range(1,8):
    UNET_ENCODE._do_register(f'efficientnet-b{i}', partial(efficientnet, name=i ) )


if __name__ == '__main__':

    encode = UNET_ENCODE.get('wrn_22')
    print(encode())

    encode = UNET_ENCODE.get('densenet121')
    print(encode())

    encode = UNET_ENCODE.get('efficientnet')
    print(encode(4))

    encode = UNET_ENCODE.get('efficientnet-b2')
    print(encode())
