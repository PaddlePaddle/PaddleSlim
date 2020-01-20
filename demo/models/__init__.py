from __future__ import absolute_import
from .mobilenet import MobileNet
from .resnet import ResNet34, ResNet50
from .mobilenet_v2 import MobileNetV2
from .pvanet import PVANet
__all__ = [
    "model_list", "MobileNet", "ResNet34", "ResNet50", "MobileNetV2", "PVANet"
]
model_list = ['MobileNet', 'ResNet34', 'ResNet50', 'MobileNetV2', 'PVANet']
