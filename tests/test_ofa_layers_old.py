# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.append("../")
import numpy as np
import unittest
import paddle
import paddle.nn as nn
from paddleslim.nas import ofa
from paddleslim.nas.ofa import OFA
from paddleslim.nas.ofa.layers_old import *


class ModelCase1(nn.Layer):
    def __init__(self):
        super(ModelCase1, self).__init__()
        models = [SuperConv2D(3, 4, 3, bias_attr=False)]
        models += [
            SuperConv2D(
                4,
                4,
                7,
                candidate_config={
                    'expand_ratio': (0.5, 1.0),
                    'kernel_size': (3, 5, 7)
                },
                transform_kernel=True)
        ]
        models += [SuperConv2D(4, 4, 3, groups=4)]
        models += [SuperConv2D(4, 4, 3, groups=2)]
        models += [SuperBatchNorm(4)]
        models += [SuperConv2DTranspose(4, 4, 3, bias_attr=False)]
        models += [
            SuperConv2DTranspose(
                4,
                4,
                7,
                candidate_config={
                    'expand_ratio': (0.5, 1.0),
                    'kernel_size': (3, 5, 7)
                },
                transform_kernel=True)
        ]
        models += [SuperConv2DTranspose(4, 4, 3, groups=4)]
        models += [SuperInstanceNorm(4)]
        models += [nn.Conv2DTranspose(4, 4, 3, groups=2)]
        models += [SuperConv2DTranspose(4, 4, 3, groups=2)]
        models += [
            SuperSeparableConv2D(
                4,
                4,
                1,
                padding=1,
                bias_attr=False,
                candidate_config={'expand_ratio': (0.5, 1.0)}),
        ]
        models += [
            SuperSeparableConv2D(
                4, 4, 1, padding=1, candidate_config={'channel': (2, 4)}),
        ]
        self.models = paddle.nn.Sequential(*models)

    def forward(self, inputs):
        return self.models(inputs)


class ModelCase2(nn.Layer):
    def __init__(self):
        super(ModelCase2, self).__init__()
        models = [
            SuperEmbedding(
                size=(64, 64), candidate_config={'expand_ratio': (0.5, 1.0)})
        ]
        models += [
            SuperLinear(
                64, 64, candidate_config={'expand_ratio': (0.5, 1.0)})
        ]
        models += [SuperLayerNorm(64)]
        models += [SuperLinear(64, 64, candidate_config={'channel': (32, 64)})]
        models += [
            SuperLinear(
                64, 64, bias_attr=False,
                candidate_config={'channel': (32, 64)})
        ]
        self.models = paddle.nn.Sequential(*models)

    def forward(self, inputs):
        return self.models(inputs)


class ModelCase3(nn.Layer):
    def __init__(self):
        super(ModelCase3, self).__init__()
        self.conv1 = SuperConv2D(
            3,
            4,
            7,
            candidate_config={'kernel_size': (3, 5, 7)},
            transform_kernel=True)
        self.conv2 = SuperConv2DTranspose(
            4,
            4,
            7,
            candidate_config={'kernel_size': (3, 5, 7)},
            transform_kernel=True)

    def forward(self, inputs):
        inputs = self.conv1(inputs, kernel_size=3)
        inputs = self.conv2(inputs, kernel_size=3)
        return inputs


class TestCase(unittest.TestCase):
    def setUp(self):
        self.model = ModelCase1()
        data_np = np.random.random((1, 3, 64, 64)).astype(np.float32)
        self.data = paddle.to_tensor(data_np)

    def test_ofa(self):
        ofa_model = OFA(self.model)
        out = self.model(self.data)


class TestCase2(TestCase):
    def setUp(self):
        self.model = ModelCase2()
        data_np = np.random.random((64, 64)).astype(np.int64)
        self.data = paddle.to_tensor(data_np)


class TestCase3(TestCase):
    def setUp(self):
        self.model = ModelCase3()
        data_np = np.random.random((1, 3, 64, 64)).astype(np.float32)
        self.data = paddle.to_tensor(data_np)


if __name__ == '__main__':
    unittest.main()
