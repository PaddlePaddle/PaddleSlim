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
import paddle.fluid as fluid
from layers import conv_bn_layer
from paddleslim.prune import StaticPruningCollections
from static_case import StaticCase


class TestPrune(StaticCase):
    def test_prune(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        #   X       X              O       X              O
        # conv1-->conv2-->sum1-->conv3-->conv4-->sum2-->conv5-->conv6
        #     |            ^ |                    ^
        #     |____________| |____________________|
        #
        # X: prune output channels
        # O: prune input channels
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
        collections = StaticPruningCollections(
            ["conv1_weights", "conv2_weights", "conv3_weights", "dummy"],
            main_program)

        params = set([
            param.name for param in main_program.all_parameters()
            if "weights" in param.name
        ])

        expected_groups = [[('conv1_weights', 0), ('conv2_weights', 1),
                            ('conv2_weights', 0), ('conv3_weights', 1),
                            ('conv4_weights', 0), ('conv5_weights', 1)],
                           [('conv3_weights', 0), ('conv4_weights', 1)]]

        self.assertTrue(len(collections._collections) == len(expected_groups))
        for _collected, _expected in zip(collections, expected_groups):
            for _info in _collected.all_pruning_details():
                _name = _info.name
                _axis = _info.axis
                if _name in params:
                    self.assertTrue((_name, _axis) in _expected)


if __name__ == '__main__':
    unittest.main()
