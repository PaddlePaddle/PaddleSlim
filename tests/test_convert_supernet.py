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
import unittest
import paddle
from paddleslim.nas.ofa.convert_super import Convert, supernet


class TestConvertSuper(unittest.TestCase):
    def setUp(self):
        self.model = paddle.vision.models.mobilenet_v1()

    def test_convert(self):
        sp_net_config = supernet(kernel_size=(3, 5, 7), expand_ratio=[1, 2, 4])
        sp_model = Convert(sp_net_config).convert(self.model)
        assert len(sp_model.sublayers()) == 151


class TestConvertSuperCase1(unittest.TestCase):
    def setUp(self):
        class Model(paddle.nn.Layer):
            def __init__(self):
                super(Model, self).__init__()
                self.fc = paddle.nn.Linear(
                    5,
                    10,
                    weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.XavierNormal()),
                    bias_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Constant(value=0.0)))

            def forward(self, inputs):
                return self.fc(inputs)

        self.model = Model()

    def test_convert(self):
        sp_net_config = supernet(expand_ratio=[1, 2, 4])
        sp_model = Convert(sp_net_config).convert(self.model)


if __name__ == '__main__':
    unittest.main()
