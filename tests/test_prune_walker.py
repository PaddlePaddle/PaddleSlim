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
import numpy as np
import paddle
import paddle.fluid as fluid
from paddleslim.prune import Pruner
from static_case import StaticCase
from layers import conv_bn_layer
import random
from paddleslim.core import GraphWrapper


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
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            conv1 = conv_bn_layer(input, 8, 3, "conv1", act='relu')
            conv2 = conv_bn_layer(conv1, 8, 3, "conv2", act='leaky_relu')
            sum1 = conv1 + conv2
            conv3 = conv_bn_layer(sum1, 8, 3, "conv3", act='relu6')
            conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
            sum2 = conv4 + sum1
            conv5 = conv_bn_layer(sum2, 8, 3, "conv5")

            flag = fluid.layers.fill_constant([1], value=1, dtype='int32')
            rand_flag = paddle.randint(2, dtype='int32')
            cond = fluid.layers.less_than(x=flag, y=rand_flag)
            cond_output = fluid.layers.create_global_var(
                shape=[1],
                value=0.0,
                dtype='float32',
                persistable=False,
                name='cond_output')

            def cond_block1():
                cond_conv = conv_bn_layer(conv5, 8, 3, "conv_cond1_1")
                fluid.layers.assign(input=cond_conv, output=cond_output)

            def cond_block2():
                cond_conv1 = conv_bn_layer(conv5, 8, 3, "conv_cond2_1")
                cond_conv2 = conv_bn_layer(cond_conv1, 8, 3, "conv_cond2_2")
                fluid.layers.assign(input=cond_conv2, output=cond_output)

            fluid.layers.cond(cond, cond_block1, cond_block2)
            sum3 = fluid.layers.sum([sum2, cond_output])

            conv6 = conv_bn_layer(sum3, 8, 3, "conv6")
            sub1 = conv6 - sum3
            mult = sub1 * sub1
            conv7 = conv_bn_layer(
                mult, 8, 3, "Depthwise_Conv7", groups=8, use_cudnn=False)
            floored = fluid.layers.floor(conv7)
            scaled = fluid.layers.scale(floored)
            concated = fluid.layers.concat([scaled, mult], axis=1)
            conv8 = conv_bn_layer(concated, 8, 3, "conv8")
            predict = fluid.layers.fc(input=conv8, size=10, act='softmax')
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            adam_optimizer = fluid.optimizer.AdamOptimizer(0.01)
            avg_cost = fluid.layers.mean(cost)
            adam_optimizer.minimize(avg_cost)

        params = []
        for param in main_program.all_parameters():
            if 'conv' in param.name:
                params.append(param.name)
        #TODO: To support pruning convolution before fc layer.
        params.remove('conv8_weights')

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(startup_program)
        x = np.random.random(size=(10, 3, 16, 16)).astype('float32')
        label = np.random.random(size=(10, 1)).astype('int64')
        loss_data, = exe.run(main_program,
                             feed={"image": x,
                                   "label": label},
                             fetch_list=[cost.name])
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

        loss_data, = exe.run(main_program,
                             feed={"image": x,
                                   "label": label},
                             fetch_list=[cost.name])


if __name__ == '__main__':
    unittest.main()
