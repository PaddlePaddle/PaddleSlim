import sys
sys.path.append("../../")
import numpy as np
import unittest
import paddle
from paddle.vision.models import MobileNetV1
import paddle.vision.transforms as T
from paddleslim.dygraph.dist import Distill, DistillConfig, add_distill_hook

s_model = MobileNetV1()
t_model = MobileNetV1()


def adaptor(model):
    mapping_keys = ['hidden', 'logits']
    mapping_layers = dict.fromkeys(mapping_keys, [])
    add_distill_hook(model, mapping_layers, ['conv1'], ['hidden'])
    add_distill_hook(model, mapping_layers, ['conv2_2'], ['hidden'])
    add_distill_hook(model, mapping_layers, ['conv3_2', 'conv4_2'],
                     ['hidden', 'hidden'])
    add_distill_hook(model, mapping_layers, ['fc'], ['logits'])
    return mapping_layers


layer_configs = [{
    'layer_S': 0,
    'layer_T': 0,
    'feature_type': 'hidden',
    'loss_function': 'l2'
}, {
    'layer_S': 1,
    'layer_T': 1,
    'feature_type': 'hidden',
    'loss_function': 'l2'
}, {
    'layer_S': 0,
    'layer_T': 0,
    'feature_type': 'logits',
    'loss_function': 'l2'
}]

distill = Distill(layer_configs, s_model, s_model, adaptor, adaptor)

transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])

train_dataset = paddle.vision.datasets.Cifar10(
    mode='train', backend='cv2', transform=transform)
val_dataset = paddle.vision.datasets.Cifar10(
    mode='test', backend='cv2', transform=transform)

place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
) else paddle.CPUPlace()
train_reader = paddle.io.DataLoader(
    train_dataset, drop_last=True, places=place, batch_size=64)
test_reader = paddle.io.DataLoader(val_dataset, places=place, batch_size=64)

adam = paddle.optimizer.Adam(
    learning_rate=0.001, parameters=s_model.parameters() + distill.parameters())

for batch_id, data in enumerate(train_reader):
    img = paddle.to_tensor(data[0])
    label = paddle.to_tensor(data[1])
    out = s_model(img)
    if batch_id == 2:
        sys.exit(0)
