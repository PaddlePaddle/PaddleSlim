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
sys.path.append("../")
import unittest
import paddle
from paddleslim import flops
from layers import conv_bn_layer
from static_case import StaticCase


class TestPrune(StaticCase):
    def test_prune(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            input = paddle.static.data(name="image", shape=[None, 3, 16, 16])
            conv1 = conv_bn_layer(input, 8, 3, "conv1")
            conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
            sum1 = conv1 + conv2
            conv3 = conv_bn_layer(sum1, 8, 3, "conv3")
            conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
            sum2 = conv4 + sum1
            conv5 = conv_bn_layer(sum2, 8, 3, "conv5")
            conv6 = conv_bn_layer(conv5, 8, 3, "conv6")
        self.assertTrue(792576 == flops(main_program))


if __name__ == '__main__':
    unittest.main()
