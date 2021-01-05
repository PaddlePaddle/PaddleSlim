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
from paddle.nn import ReLU
from paddleslim.nas import ofa
from paddleslim.nas.ofa import OFA, RunConfig, DistillConfig
from paddleslim.nas.ofa.convert_super import supernet
from paddleslim.nas.ofa.layers import *


class ModelCase1(nn.Layer):
    def __init__(self):
        super(ModelCase1, self).__init__()
        models = [SuperConv2D(3, 4, 3, bias_attr=False)]
        models += [SuperConv2D(4, 4, 3, groups=4)]
        models += [SuperConv2D(4, 4, 3, groups=2)]
        models += [SuperConv2DTranspose(4, 4, 3, bias_attr=False)]
        models += [SuperConv2DTranspose(4, 4, 3, groups=4)]
        models += [nn.Conv2DTranspose(4, 4, 3, groups=2)]
        models += [SuperConv2DTranspose(4, 4, 3, groups=2)]
        models += [
            SuperSeparableConv2D(
                4,
                4,
                1,
                padding=1,
                bias_attr=False,
                candidate_config={'expand_ratio': (1.0, 2.0)}),
        ]
        self.models = paddle.nn.Sequential(*models)

    def forward(self, inputs):
        return self.models(inputs)


class TestCase(unittest.TestCase):
    def setUp(self):
        self.model = ModelCase1()
        data_np = np.random.random((1, 3, 64, 64)).astype(np.float32)
        self.data = paddle.to_tensor(data_np)

    def test_ofa(self):
        ofa_model = OFA(self.model)
        out = self.model(self.data)


if __name__ == '__main__':
    unittest.main()
