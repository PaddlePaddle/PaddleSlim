# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
from paddleslim.prune import AutoPruner
from static_case import StaticCase
from layers import conv_bn_layer


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

        shapes = {}
        params = []
        for param in main_program.global_block().all_parameters():
            shapes[param.name] = param.shape
            if 'weights' in param.name:
                params.append(param.name)

        val_program = fluid.default_main_program().clone(for_test=True)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        scope = fluid.Scope()
        exe.run(startup_program, scope=scope)

        pruner = AutoPruner(
            val_program,
            fluid.global_scope(),
            place,
            params=params,
            init_ratios=[0.33] * len(params),
            pruned_flops=0.5,
            pruned_latency=None,
            server_addr=("", 0),
            init_temperature=100,
            reduce_rate=0.85,
            max_try_times=300,
            max_client_num=10,
            search_steps=100,
            max_ratios=0.9,
            min_ratios=0.,
            is_server=True,
            key="auto_pruner")
        baseratio = None
        lastratio = None
        for i in range(10):
            pruned_program, pruned_val_program = pruner.prune(
                fluid.default_main_program(), val_program)
            score = 0.2
            pruner.reward(score)
            if i == 0:
                baseratio = pruner._current_ratios
            if i == 9:
                lastratio = pruner._current_ratios
        changed = False
        for i in range(len(baseratio)):
            if baseratio[i] != lastratio[i]:
                changed = True
        self.assertTrue(changed == True)


if __name__ == '__main__':
    unittest.main()
