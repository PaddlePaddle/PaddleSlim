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
from paddleslim.prune import Pruner
from layers import conv_bn_layer


class TestPrune(unittest.TestCase):
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

        shapes = {}
        for param in main_program.global_block().all_parameters():
            shapes[param.name] = param.shape

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        scope = fluid.Scope()
        exe.run(startup_program, scope=scope)
        pruner = Pruner()
        main_program = pruner.prune(
            main_program,
            scope,
            params=["conv4_weights"],
            ratios=[0.5],
            place=place,
            lazy=False,
            only_graph=False,
            param_backup=None,
            param_shape_backup=None)

        shapes = {
            "conv1_weights": (4L, 3L, 3L, 3L),
            "conv2_weights": (4L, 4L, 3L, 3L),
            "conv3_weights": (8L, 4L, 3L, 3L),
            "conv4_weights": (4L, 8L, 3L, 3L),
            "conv5_weights": (8L, 4L, 3L, 3L),
            "conv6_weights": (8L, 8L, 3L, 3L)
        }

        for param in main_program.global_block().all_parameters():
            if "weights" in param.name:
                self.assertTrue(param.shape == shapes[param.name])


if __name__ == '__main__':
    unittest.main()
