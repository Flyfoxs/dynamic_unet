import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from file_cache import *
from fastai.vision import *
from fastai.callbacks.hooks import *
from fvcore.common.registry import Registry
from torchvision import models
from enum import Enum
from fastai.vision.models.unet import UnetBlock


def dummy_eval(m: nn.Module, size: tuple = (64, 64)):
    "Pass a `dummy_batch` in evaluation mode in `m` with `size`."
    dummy = torch.rand((1, 3, *size)).requires_grad_(False)
    return m.eval()(dummy)


def in_channels(m: nn.Module) -> List[int]:
    "Return the shape of the first weight layer in `m`."
    for l in m.modules():
        if hasattr(l, 'weight'): return l.weight.shape[1]
    raise Exception(f'No weight layer:{type(m)}')


def flatten_moduleList(module: nn.Module) -> List[nn.Module]:
    "If the ModuleList can be found in children, flatten it. Since ModuleList can not support hook "
    res_list = []
    for item in module.children():
        if isinstance(item, nn.ModuleList):
            res_list.extend(flatten_moduleList(item))
        else:
            res_list.append(item)
    return res_list

def get_unet_config(model, img_size=(512, 512)):
    "Cut the network to several blocks, the width and high of the image are reduced by half. And the image W and H >= 7"
    x = torch.rand(1, in_channels(model), *img_size)
    hooks = []
    count = 0
    layer_meta = []
    layers = []


    def hook(module, input, output):
        "To get the meta of the layer infomation"
        nonlocal count
        if len(output.shape) == 4:
            b, c, w, h = output.shape
            layer_meta.append((count, type(module).__name__, c, w, h, output.shape))
        layers.append(module)
        count += 1

    for module in flatten_moduleList(model):
        hooks.append(module.register_forward_hook(hook))

    # make a forward pass to trigger the hooks
    model(x)
    for h in hooks:
        h.remove()

    layer_meta = pd.DataFrame(layer_meta, columns=['sn', 'layer', 'c', 'w', 'h', 'size'])
    img_size = [x.shape[-1] // (2 ** i) for i in range(8)]
    img_size = [size for size in img_size if size >= 7]
    layer_meta:pd.DataFrame = layer_meta.loc[(layer_meta.h.isin(img_size))].drop_duplicates(['h'], keep='last')
    layer_meta = layer_meta.head(5)
    print(layer_meta)
    assert len(layer_meta) == 5, f'Only cut {len(layer_meta)} layers from the pretrained model '

    layer_size = list(layer_meta['size'])
    layers = [layers[i] for i in layer_meta.sn]
    return layer_size, layers


class DynamicUnet(SequentialEx):
    "Create a U-Net from a given architecture."

    def __init__(self, encoder: nn.Module, n_classes: int, img_size: Tuple[int, int] = (256, 256),
                 blur: bool = False,
                 blur_final=True, self_attention: bool = False,
                 y_range: Optional[Tuple[float, float]] = None,
                 last_cross: bool = True, bottle: bool = False, **kwargs):


        imsize = tuple(img_size)
        sfs_szs, select_layer = get_unet_config(encoder, img_size)
        ni = sfs_szs[-1][1]
        sfs_szs = list(reversed(sfs_szs[:-1]))
        select_layer = list(reversed(select_layer[:-1]))
        self.sfs = hook_outputs(select_layer, detach=False)
        x = dummy_eval(encoder, imsize).detach()

        middle_conv = nn.Sequential(conv_layer(ni, ni * 2, **kwargs),
                                    conv_layer(ni * 2, ni, **kwargs)).eval()
        x = middle_conv(x)
        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

        for i, x_size in enumerate(sfs_szs):
            not_final = i != len(sfs_szs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(x_size[1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i == len(sfs_szs) - 3)
            unet_block = UnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final, blur=do_blur, self_attention=sa,
                                   **kwargs).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]: layers.append(PixelShuffle_ICNR(ni, **kwargs))
        x = PixelShuffle_ICNR(ni)(x)
        if imsize != x.shape[-2:]: layers.append(Lambda(lambda x: F.interpolate(x, imsize, mode='nearest')))
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(res_block(ni, bottle=bottle, **kwargs))
        layers += [conv_layer(ni, n_classes, ks=1, use_activ=False, **kwargs)]
        if y_range is not None: layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)


def efficient_unet(name='5'):
    from efficientnet_pytorch import EfficientNet
    class EfficientNet_(EfficientNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, inputs):
            x = self.extract_features(inputs)
            return x
    return EfficientNet_.from_pretrained(f'efficientnet-b{name}', in_channels=3)


if __name__ == '__main__':
    encoder = efficient_unet()
    unet = to_device(
        DynamicUnet(encoder, n_classes=5, img_size=(224, 224), blur=False, blur_final=False,
                    self_attention=False, y_range=None, norm_type=NormType,
                    last_cross=True,
                    bottle=False), 'cuda')
