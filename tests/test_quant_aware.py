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
import paddle.fluid as fluid
from paddleslim.quant import quant_aware, convert
from static_case import StaticCase
sys.path.append("../demo")
from models import MobileNet
from layers import conv_bn_layer
import paddle.dataset.mnist as reader
from paddle.fluid.framework import IrGraph
from paddle.fluid import core
import numpy as np


class TestQuantAwareCase1(StaticCase):
    def get_model(self):
        image = fluid.layers.data(
            name='image', shape=[1, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        model = MobileNet()
        out = model.net(input=image, class_dim=10)
        cost = fluid.layers.cross_entropy(input=out, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        startup_prog = fluid.default_startup_program()
        train_prog = fluid.default_main_program()
        return startup_prog, train_prog

    def get_op_number(self, prog):

        graph = IrGraph(core.Graph(prog.desc), for_test=False)
        quant_op_nums = 0
        op_nums = 0
        for op in graph.all_op_nodes():
            if op.name() in ['conv2d', 'depthwise_conv2d', 'mul']:
                op_nums += 1
            elif 'fake_' in op.name():
                quant_op_nums += 1
        return op_nums, quant_op_nums

    def test_quant_op(self):
        startup_prog, train_prog = self.get_model()
        place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        config_1 = {
            'weight_quantize_type': 'channel_wise_abs_max',
            'activation_quantize_type': 'moving_average_abs_max',
            'quantize_op_types': ['depthwise_conv2d', 'mul', 'conv2d'],
        }

        quant_prog_1 = quant_aware(
            train_prog, place, config=config_1, for_test=True)
        op_nums_1, quant_op_nums_1 = self.get_op_number(quant_prog_1)
        convert_prog_1 = convert(quant_prog_1, place, config=config_1)
        convert_op_nums_1, convert_quant_op_nums_1 = self.get_op_number(
            convert_prog_1)

        config_1['not_quant_pattern'] = ['last_fc']
        quant_prog_2 = quant_aware(
            train_prog, place, config=config_1, for_test=True)
        op_nums_2, quant_op_nums_2 = self.get_op_number(quant_prog_2)
        convert_prog_2 = convert(quant_prog_2, place, config=config_1)
        convert_op_nums_2, convert_quant_op_nums_2 = self.get_op_number(
            convert_prog_2)

        self.assertTrue(op_nums_1 == op_nums_2)
        # test quant_aware op numbers
        self.assertTrue(op_nums_1 * 4 == quant_op_nums_1)
        # test convert op numbers
        self.assertTrue(convert_op_nums_1 * 2 == convert_quant_op_nums_1)
        # test skip_quant
        self.assertTrue(quant_op_nums_1 - 4 == quant_op_nums_2)
        self.assertTrue(convert_quant_op_nums_1 - 2 == convert_quant_op_nums_2)


class TestQuantAwareCase2(StaticCase):
    def test_accuracy(self):
        image = fluid.layers.data(
            name='image', shape=[1, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        model = MobileNet()
        out = model.net(input=image, class_dim=10)
        cost = fluid.layers.cross_entropy(input=out, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
        optimizer = fluid.optimizer.Momentum(
            momentum=0.9,
            learning_rate=0.01,
            regularization=fluid.regularizer.L2Decay(4e-5))
        optimizer.minimize(avg_cost)
        main_prog = fluid.default_main_program()
        val_prog = main_prog.clone(for_test=True)

        place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        feeder = fluid.DataFeeder([image, label], place, program=main_prog)
        train_reader = paddle.fluid.io.batch(
            paddle.dataset.mnist.train(), batch_size=64)
        eval_reader = paddle.fluid.io.batch(
            paddle.dataset.mnist.test(), batch_size=64)

        def train(program):
            iter = 0
            for data in train_reader():
                cost, top1, top5 = exe.run(
                    program,
                    feed=feeder.feed(data),
                    fetch_list=[avg_cost, acc_top1, acc_top5])
                iter += 1
                if iter % 100 == 0:
                    print(
                        'train iter={}, avg loss {}, acc_top1 {}, acc_top5 {}'.
                        format(iter, cost, top1, top5))

        def test(program):
            iter = 0
            result = [[], [], []]
            for data in eval_reader():
                cost, top1, top5 = exe.run(
                    program,
                    feed=feeder.feed(data),
                    fetch_list=[avg_cost, acc_top1, acc_top5])
                iter += 1
                if iter % 100 == 0:
                    print('eval iter={}, avg loss {}, acc_top1 {}, acc_top5 {}'.
                          format(iter, cost, top1, top5))
                result[0].append(cost)
                result[1].append(top1)
                result[2].append(top5)
            print(' avg loss {}, acc_top1 {}, acc_top5 {}'.format(
                np.mean(result[0]), np.mean(result[1]), np.mean(result[2])))
            return np.mean(result[1]), np.mean(result[2])

        train(main_prog)
        top1_1, top5_1 = test(main_prog)

        config = {
            'weight_quantize_type': 'channel_wise_abs_max',
            'activation_quantize_type': 'moving_average_abs_max',
            'quantize_op_types': ['depthwise_conv2d', 'mul', 'conv2d'],
        }
        quant_train_prog = quant_aware(main_prog, place, config, for_test=False)
        quant_eval_prog = quant_aware(val_prog, place, config, for_test=True)
        train(quant_train_prog)
        quant_eval_prog, int8_prog = convert(
            quant_eval_prog, place, config, save_int8=True)
        top1_2, top5_2 = test(quant_eval_prog)
        # values before quantization and after quantization should be close
        print("before quantization: top1: {}, top5: {}".format(top1_1, top5_1))
        print("after quantization: top1: {}, top5: {}".format(top1_2, top5_2))


if __name__ == '__main__':
    unittest.main()
