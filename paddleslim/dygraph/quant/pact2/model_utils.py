import paddle
from paddle.nn import Conv2D, Conv2DTranspose, Linear, BatchNorm2D

def is_conv_deconv_linear(layer):
    return isinstance(layer, (Conv2D, Conv2DTranspose, Linear))

def is_conv(layer):
    return isinstance(layer, Conv2D)

def is_deconv(layer):
    return isinstance(layer, Conv2DTranspose)

def is_dwconv(layer):
    return is_conv(layer) and (layer.weight.size(1) == 1)

def is_linear(layer):
    return isinstance(layer, Linear)

