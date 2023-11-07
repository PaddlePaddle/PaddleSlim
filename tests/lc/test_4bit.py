import sys
sys.path.append("../../")
import numpy as np
import unittest
import paddle
from paddleslim.lc.layers import NF4Linear, FP4Linear

class TransformerEncoderLayer(paddle.nn.Layer):
    def __init__(self):
        super(TransformerEncoderLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(64, 128)
        self.linear2 = paddle.nn.Linear(128, 64)
        self.norm1 = paddle.nn.LayerNorm(64, epsilon=1e-12)
        self.norm2 = paddle.nn.LayerNorm(64, epsilon=1e-12)
        self.activation = paddle.nn.ReLU()
        self.d_model = 64
        self.nhead = 2

    def forward(self, src, src_mask=None, rand_mask_idx=None, query_mask=None, key_mask=None):
        src = self.norm1(src)
        src = self.norm2(src)
        src = self.linear2(self.activation(self.linear1(src)))

        return src

def split_prefix(net, name_list):
    if len(name_list) > 1:
        net = self.split_prefix(getattr(net, name_list[0]), name_list[1:])
    elif len(name_list) == 1:
        net = getattr(net, name_list[0])
    else:
        raise NotImplementedError("name error")
    return net

class TestNF4(unittest.TestCase):
    def setUp(self):
        self.model = TransformerEncoderLayer()

    def replace_linear(self, model):
        replace_layers = {}
        for name, child in model.named_sublayers():
            if isinstance(child, paddle.nn.Linear):
                replace_layers[name] = NF4Linear(child, use_double_quant=True)

        for key, value in replace_layers.items():
            name_list = key.split('.')
            if len(name_list) > 1:
                net = split_prefix(model, name_list[:-1])
            else:
                net = model
            setattr(net, name_list[-1], value)

    def test_nf4_blockwise(self):
        self.replace_linear(self.model)
        for name, child in self.model.named_sublayers():
            if isinstance(child, NF4Linear):
                child.quantize()

        src = paddle.rand((2, 4, 64))
        self.model(src)

class TestFP4(unittest.TestCase):
    def setUp(self):
        self.model = TransformerEncoderLayer()

    def replace_linear(self, model):
        replace_layers = {}
        for name, child in model.named_sublayers():
            if isinstance(child, paddle.nn.Linear):
                replace_layers[name] = FP4Linear(child, use_double_quant=True)

        for key, value in replace_layers.items():
            name_list = key.split('.')
            if len(name_list) > 1:
                net = split_prefix(model, name_list[:-1])
            else:
                net = model
            setattr(net, name_list[-1], value)

    def test_fp4_blockwise(self):
        self.replace_linear(self.model)
        for name, child in self.model.named_sublayers():
            if isinstance(child, FP4Linear):
                child.quantize()

        src = paddle.rand((2, 4, 64))
        self.model(src)

if __name__ == '__main__':
    unittest.main()
