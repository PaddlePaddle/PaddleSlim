from __future__ import absolute_import
from .mobilenet import MobileNet, MobileNetCifar, MobileNetSensitive30, MobileNetSensitive50
from .resnet import ResNet34, ResNet50
from .resnet_vd import ResNet50_vd, ResNet101_vd
from .mobilenet_v2 import MobileNetV2_x0_25, MobileNetV2
from .pvanet import PVANet
from .slimfacenet import SlimFaceNet_A_x0_60, SlimFaceNet_B_x0_75, SlimFaceNet_C_x0_75
from .mobilenet_v3 import *
__all__ = [
    "model_list", "MobileNet", "ResNet34", "MobileNetSensitive30",
    "MobileNetSensitive50", "MobileNetCifar", "ResNet50", "MobileNetV2",
    "PVANet", "ResNet50_vd", "ResNet101_vd", "MobileNetV2_x0_25"
]
model_list = [
    'MobileNet', 'ResNet34', 'ResNet50', 'MobileNetV2', 'MobileNetSensitive30',
    'MobileNetSensitive50'
    'MobileNetCifar', 'PVANet', 'ResNet50_vd', "ResNet101_vd",
    "MobileNetV2_x0_25"
]

__all__ += mobilenet_v3.__all__
model_list += mobilenet_v3.__all__
