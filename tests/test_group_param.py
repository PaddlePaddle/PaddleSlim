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
from paddleslim.prune import collect_convs
from static_case import StaticCase


class TestPrune(StaticCase):
    def test_prune(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        #   X       X              O       X              O
        # conv1-->conv2-->sum1-->conv3-->conv4-->sum2-->conv5-->conv6
        #     |            ^ |                    ^
        #     |____________| |____________________|
        #
        # X: prune output channels
        # O: prune input channels
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
        collected_groups = collect_convs(
            ["conv1_weights", "conv2_weights", "conv3_weights", "dummy"],
            main_program)
        while [] in collected_groups:
            collected_groups.remove([])
        print(collected_groups)

        params = set([
            param.name for param in main_program.all_parameters()
            if "weights" in param.name
        ])

        expected_groups = [[('conv1_weights', 0), ('conv2_weights', 1),
                            ('conv2_weights', 0), ('conv3_weights', 1),
                            ('conv4_weights', 0), ('conv5_weights', 1)],
                           [('conv3_weights', 0), ('conv4_weights', 1)]]

        self.assertTrue(len(collected_groups) == len(expected_groups))
        for _collected, _expected in zip(collected_groups, expected_groups):
            for _name, _axis, _ in _collected:
                if _name in params:
                    self.assertTrue((_name, _axis) in _expected)
            for _name, _axis in _expected:
                if _name in params:
                    self.assertTrue((_name, _axis, []) in _collected)


if __name__ == '__main__':
    unittest.main()
