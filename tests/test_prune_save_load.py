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
from paddleslim.prune import Pruner, save_model, load_model
from layers import conv_bn_layer
import numpy as np
import numpy


class TestSaveAndLoad(unittest.TestCase):
    def test_prune(self):
        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            input = fluid.data(name="image", shape=[None, 3, 16, 16])
            conv1 = conv_bn_layer(input, 8, 3, "conv1")
            conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
            sum1 = conv1 + conv2
            conv3 = conv_bn_layer(sum1, 8, 3, "conv3")
            conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
            sum2 = conv4 + sum1
            conv5 = conv_bn_layer(sum2, 8, 3, "conv5")
            conv6 = conv_bn_layer(conv5, 8, 3, "conv6")

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)


        scope = fluid.global_scope()
        exe.run(startup_program, scope=scope)
        criterion = 'bn_scale'
        pruner = Pruner(criterion)
        main_program, _, _ = pruner.prune(
            train_program,
            scope,
            params=["conv4_weights"],
            ratios=[0.5],
            place=place,
            lazy=False,
            only_graph=False,
            param_backup=None,
            param_shape_backup=None)

        x = numpy.random.random(size=(10, 3, 16, 16)).astype('float32')
        loss_data, = exe.run(train_program,
                            feed={"image": x},
                            fetch_list=[conv6.name])
        
        save_model(exe, main_program, 'model_file')
        pruned_program = fluid.Program()
        pruned_startup_program = fluid.Program()
        with fluid.program_guard(pruned_program, pruned_startup_program):
            input = fluid.data(name="image", shape=[None, 3, 16, 16])
            conv1 = conv_bn_layer(input, 8, 3, "conv1")
            conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
            sum1 = conv1 + conv2
            conv3 = conv_bn_layer(sum1, 8, 3, "conv3")
            conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
            sum2 = conv4 + sum1
            conv5 = conv_bn_layer(sum2, 8, 3, "conv5")
            conv6 = conv_bn_layer(conv5, 8, 3, "conv6")
        exe.run(pruned_startup_program)
        load_model(exe, pruned_program, 'model_file')
        shapes = {
            "conv1_weights": (4, 3, 3, 3),
            "conv2_weights": (4, 4, 3, 3),
            "conv3_weights": (8, 4, 3, 3),
            "conv4_weights": (4, 8, 3, 3),
            "conv5_weights": (8, 4, 3, 3),
            "conv6_weights": (8, 8, 3, 3)
        }

        for param in pruned_program.global_block().all_parameters():
            if "weights" in param.name:
                print("param: {}; param shape: {}".format(param.name,
                                                          param.shape))
                self.assertTrue(param.shape == shapes[param.name])


if __name__ == '__main__':
    unittest.main()
