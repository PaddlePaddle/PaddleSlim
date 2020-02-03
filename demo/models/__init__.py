from __future__ import absolute_import
from .mobilenet import MobileNet
from .resnet import ResNet34, ResNet50
from .resnet_vd import ResNet50_vd
from .mobilenet_v2 import MobileNetV2
from .pvanet import PVANet
__all__ = [
    "model_list", "MobileNet", "ResNet34", "ResNet50", "MobileNetV2", "PVANet",
    "ResNet50_vd"
]
model_list = [
    'MobileNet', 'ResNet34', 'ResNet50', 'MobileNetV2', 'PVANet', 'ResNet50_vd'
]
