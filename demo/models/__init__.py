from __future__ import absolute_import
from .mobilenet import MobileNet
from .resnet import ResNet34, ResNet50
from .resnet_vd import ResNet50_vd
from .mobilenet_v2 import MobileNetV2
from .pvanet import PVANet
from .mobilenet_v3 import *
__all__ = [
    "model_list", "MobileNet", "ResNet34", "ResNet50", "MobileNetV2", "PVANet",
    "ResNet50_vd"
]
model_list = [
    'MobileNet', 'ResNet34', 'ResNet50', 'MobileNetV2', 'PVANet',
    'ResNet50_vd', 'MobileNetV3', 'MobileNetV3_small_x0_25',
    'MobileNetV3_small_x0_5', 'MobileNetV3_small_x0_75',
    'MobileNetV3_small_x1_0', 'MobileNetV3_small_x1_25',
    'MobileNetV3_large_x0_25', 'MobileNetV3_large_x0_5',
    'MobileNetV3_large_x0_75', 'MobileNetV3_large_x1_0',
    'MobileNetV3_large_x1_25'
]
