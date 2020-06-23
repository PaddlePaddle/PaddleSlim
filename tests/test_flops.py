# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import numpy as np
sys.path.append("../")
import unittest
import paddle.fluid as fluid
from paddleslim.analysis import flops, dygraph_flops
from paddle.fluid.dygraph.base import to_variable
from layers import conv_bn_layer


class TestFLOPs(unittest.TestCase):
    def test_static_flops(self):
        fluid.disable_dygraph()
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            input = fluid.data(name="image", shape=[None, 3, 16, 16])
            conv1 = conv_bn_layer(input, 8, 3, "conv1")
            conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
            sum1 = conv1 + conv2
            conv3 = conv_bn_layer(sum1, 8, 3, "conv3")
            conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
            sum2 = conv4 + sum1
            conv5 = conv_bn_layer(sum2, 8, 3, "conv5")
            conv6 = conv_bn_layer(conv5, 8, 3, "conv6")
        self.assertTrue(792576 == flops(main_program))

    def test_dygraph_flops(self):
        fluid.enable_dygraph()
        model = Net()
        data_np = np.random.random((1, 3, 16, 16)).astype('float32')
        data = to_variable(data_np)
        self.assertTrue(125280 == dygraph_flops(model, (data)))


class Net(fluid.dygraph.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = fluid.dygraph.Conv2D(3, 8, 3, padding=1)
        self.bn1 = fluid.dygraph.BatchNorm(8)
        self.conv2 = fluid.dygraph.Conv2DTranspose(8, 3, 3)
        self.bn2 = fluid.dygraph.BatchNorm(3)

    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = self.bn2(self.conv2(y))
        return y


if __name__ == '__main__':
    unittest.main()
