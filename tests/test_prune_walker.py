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
            conv1 = conv_bn_layer(input, 8, 3, "conv1", act='relu')
            conv2 = conv_bn_layer(conv1, 8, 3, "conv2", act='leaky_relu')
            sum1 = conv1 + conv2
            conv3 = conv_bn_layer(sum1, 8, 3, "conv3", act='relu6')
            conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
            sum2 = conv4 + sum1
            conv5 = conv_bn_layer(sum2, 8, 3, "conv5")
            sum3 = fluid.layers.sum([sum2, conv5])
            conv6 = conv_bn_layer(sum3, 8, 3, "conv6")
            sub1 = conv6 - sum3
            mult = sub1 * sub1
            conv7 = conv_bn_layer(mult, 8, 3, "Depthwise_Conv7", groups=8)
            floored = fluid.layers.floor(conv7)
            scaled = fluid.layers.scale(floored)
            concated = fluid.layers.concat([scaled, mult], axis=1)

        params = []
        for param in main_program.all_parameters():
            if 'conv' in param.name:
                params.append(param.name)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        pruner = Pruner()
        main_program, _, _ = pruner.prune(
            main_program,
            fluid.global_scope(),
            params=params,
            ratios=[0.5] * len(params),
            place=place,
            lazy=False,
            only_graph=False,
            param_backup=None,
            param_shape_backup=None)


if __name__ == '__main__':
    unittest.main()
