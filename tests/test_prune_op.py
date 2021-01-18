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
from static_case import StaticCase
import paddle.fluid as fluid
from paddleslim.prune import Pruner
from static_case import StaticCase
from layers import conv_bn_layer


class TestPrune(StaticCase):
    def test_concat(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        #                                  X              
        # conv1   conv2-->concat         conv3-->sum-->out
        #     |            ^ |                    ^
        #     |____________| |____________________|
        #
        with fluid.program_guard(main_program, startup_program):
            input = fluid.data(name="image", shape=[None, 3, 16, 16])
            conv1 = conv_bn_layer(input, 8, 3, "conv1")
            conv2 = conv_bn_layer(input, 8, 3, "conv2", sync_bn=True)
            tmp = fluid.layers.concat([conv1, conv2], axis=1)
            conv3 = conv_bn_layer(input, 16, 3, "conv3", bias=None)
            out = conv3 + tmp

        shapes = {}
        for param in main_program.global_block().all_parameters():
            shapes[param.name] = param.shape

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        scope = fluid.Scope()
        exe.run(startup_program, scope=scope)
        pruner = Pruner()
        # test backward search of concat
        pruned_program, _, _ = pruner.prune(
            main_program,
            scope,
            params=["conv3_weights"],
            ratios=[0.5],
            place=place,
            lazy=False,
            only_graph=True,
            param_backup=None,
            param_shape_backup=None)
        shapes = {
            "conv3_weights": (8, 3, 3, 3),
            "conv2_weights": (4, 3, 3, 3),
            "conv1_weights": (4, 3, 3, 3)
        }
        for param in pruned_program.global_block().all_parameters():
            if "weights" in param.name and "conv2d" in param.name:
                self.assertTrue(shapes[param.name] == param.shape)

        # test forward search of concat
        pruned_program, _, _ = pruner.prune(
            main_program,
            scope,
            params=["conv1_weights", "conv2_weights"],
            ratios=[0.5, 0.5],
            place=place,
            lazy=False,
            only_graph=False,
            param_backup=None,
            param_shape_backup=None)

        shapes = {
            "conv1_weights": (4, 3, 3, 3),
            "conv1_bn_scale": (4, ),
            "conv1_bn_variance": (4, ),
            "conv1_bn_mean": (4, ),
            "conv1_bn_offset": (4, ),
            "conv2_weights": (4, 3, 3, 3),
            "sync_batch_norm_0.w_0": (4, ),
            "sync_batch_norm_0.w_1": (4, ),
            "conv2_bn_scale": (4, ),
            "conv2_bn_offset": (4, ),
            "conv3_weights": (8, 3, 3, 3),
            "conv3_bn_mean": (8, ),
            "conv3_bn_offset": (8, ),
            "conv3_bn_scale": (8, ),
            "conv3_bn_variance": (8, ),
            "conv3_out.b_0": (8, ),
        }

        for param in pruned_program.global_block().all_parameters():
            if "weights" in param.name and "conv2d" in param.name:
                self.assertTrue(shapes[param.name] == param.shape)


if __name__ == '__main__':
    unittest.main()
