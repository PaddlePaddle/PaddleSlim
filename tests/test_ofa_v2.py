# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddleslim.nas.ofa.convert_super import Convert, supernet


class ModelV1(nn.Layer):
    def __init__(self, name=''):
        super(ModelV1, self).__init__()
        self.model = nn.Sequential(nn.Conv2D(3, 12, 16), nn.ReLU())
        self.cls = self.create_parameter(
            attr=paddle.ParamAttr(
                name=name + 'cls',
                initializer=nn.initializer.Assign(
                    paddle.zeros(shape=(2, 12, 17, 17)))),
            shape=(2, 12, 17, 17))

    def forward(self, inputs):
        return self.cls + self.model(inputs)


class ModelShortcut(nn.Layer):
    def __init__(self):
        super(ModelShortcut, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2D(3, 12, 1), nn.BatchNorm2D(12), nn.ReLU())
        self.branch1 = nn.Sequential(
            nn.Conv2D(12, 12, 1),
            nn.BatchNorm2D(12),
            nn.ReLU(),
            nn.Conv2D(
                12, 12, 1, groups=12),
            nn.BatchNorm2D(12),
            nn.ReLU(),
            nn.Conv2D(
                12, 12, 1, groups=12),
            nn.BatchNorm2D(12),
            nn.ReLU())
        self.branch2 = nn.Sequential(
            nn.Conv2D(12, 12, 1),
            nn.BatchNorm2D(12),
            nn.ReLU(),
            nn.Conv2D(
                12, 12, 1, groups=12),
            nn.BatchNorm2D(12),
            nn.ReLU(),
            nn.Conv2D(12, 12, 1),
            nn.BatchNorm2D(12),
            nn.ReLU())
        self.out = nn.Sequential(
            nn.Conv2D(12, 12, 1), nn.BatchNorm2D(12), nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        y = self.branch1(x)
        y = x + y
        z = self.branch2(y)
        z = z + y
        z = self.out(z)
        return z


class ModelElementwise(nn.Layer):
    def __init__(self):
        super(ModelElementwise, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2D(3, 12, 1), nn.BatchNorm2D(12), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2D(12, 24, 3), nn.BatchNorm2D(24), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2D(24, 12, 1), nn.BatchNorm2D(12), nn.ReLU())
        self.out = nn.Sequential(
            nn.Conv2D(12, 6, 1), nn.BatchNorm2D(6), nn.ReLU())

    def forward(self, x):
        d = paddle.randn(shape=[2, 12, x.shape[2], x.shape[3]], dtype='float32')
        d = nn.functional.softmax(d)
        x = self.conv1(x)
        x = x + d
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.out(x)
        return x


class TestOFAV2(unittest.TestCase):
    def setUp(self):
        model = ModelV1()
        sp_net_config = supernet(expand_ratio=[0.25, 0.5, 1.0])
        self.model = Convert(sp_net_config).convert(model)
        self.images = paddle.randn(shape=[2, 3, 32, 32], dtype='float32')

    def test_ofa(self):
        self.ofa_model = OFA(self.model)
        self.ofa_model.set_epoch(0)
        self.ofa_model.set_task('expand_ratio')
        out, _ = self.ofa_model(self.images)
        print(self.ofa_model.get_current_config)


class TestOFAV2Export(unittest.TestCase):
    def setUp(self):
        model = ModelV1(name='export')
        sp_net_config = supernet(expand_ratio=[0.25, 0.5, 1.0])
        self.model = Convert(sp_net_config).convert(model)
        self.images = paddle.randn(shape=[2, 3, 32, 32], dtype='float32')
        self.ofa_model = OFA(self.model)

    def test_export(self):
        origin_model = ModelV1(name='origin')
        net_config = {'model.0': {}}
        self.ofa_model.export(
            net_config,
            input_shapes=[1, 3, 32, 32],
            input_dtypes=['float32'],
            origin_model=origin_model)


class Testelementwise(unittest.TestCase):
    def setUp(self):
        model = ModelElementwise()
        sp_net_config = supernet(expand_ratio=[0.25, 0.5, 1.0])
        self.model = Convert(sp_net_config).convert(model)
        self.images = paddle.randn(shape=[2, 3, 32, 32], dtype='float32')

    def test_ofa(self):
        self.ofa_model = OFA(self.model)
        self.ofa_model.set_epoch(0)
        self.ofa_model.set_task('expand_ratio')
        out, _ = self.ofa_model(self.images)
        assert list(self.ofa_model._ofa_layers.keys(
        )) == ['conv2.0', 'conv3.0', 'out.0']


class TestShortcutSkiplayers(unittest.TestCase):
    def setUp(self):
        model = ModelShortcut()
        sp_net_config = supernet(expand_ratio=[0.5, 1.0])
        self.model = Convert(sp_net_config).convert(model)
        self.images = paddle.randn(shape=[2, 3, 32, 32], dtype='float32')
        self.init_config()
        self.ofa_model = OFA(self.model, run_config=self.run_config)
        self.ofa_model._clear_search_space(self.images)

    def init_config(self):
        default_run_config = {'skip_layers': ['branch1.6']}
        self.run_config = RunConfig(**default_run_config)

    def test_shortcut(self):
        self.ofa_model.set_epoch(0)
        self.ofa_model.set_task('expand_ratio')
        for i in range(5):
            self.ofa_model(self.images)
        assert list(self.ofa_model._ofa_layers.keys()) == ['branch2.0', 'out.0']


class TestShortcutSkiplayersCase1(TestShortcutSkiplayers):
    def init_config(self):
        default_run_config = {'skip_layers': ['conv1.0']}
        self.run_config = RunConfig(**default_run_config)


class TestShortcutSkiplayersCase2(TestShortcutSkiplayers):
    def init_config(self):
        default_run_config = {'skip_layers': ['branch2.0']}
        self.run_config = RunConfig(**default_run_config)

    def test_shortcut(self):
        assert list(self.ofa_model._ofa_layers.keys()) == ['conv1.0', 'out.0']


if __name__ == '__main__':
    unittest.main()
