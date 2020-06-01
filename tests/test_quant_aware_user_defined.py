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
sys.path.append("../demo")
from models import MobileNet
from layers import conv_bn_layer
import paddle.dataset.mnist as reader
from paddle.fluid.framework import IrGraph
from paddle.fluid import core
import numpy as np

from paddle.fluid.layer_helper import LayerHelper


def pact(x, name=None):
    helper = LayerHelper("pact", **locals())
    dtype = 'float32'
    init_thres = 20
    u_param_attr = fluid.ParamAttr(
        name=x.name + '_pact',
        initializer=fluid.initializer.ConstantInitializer(value=init_thres),
        regularizer=fluid.regularizer.L2Decay(0.0001),
        learning_rate=1)
    u_param = helper.create_parameter(
        attr=u_param_attr, shape=[1], dtype=dtype)
    x = fluid.layers.elementwise_sub(
        x, fluid.layers.relu(fluid.layers.elementwise_sub(x, u_param)))
    x = fluid.layers.elementwise_add(
        x, fluid.layers.relu(fluid.layers.elementwise_sub(-u_param, x)))

    return x


def get_optimizer():
    return fluid.optimizer.MomentumOptimizer(0.0001, 0.9)


class TestQuantAwareCase1(unittest.TestCase):
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

    def test_accuracy(self):
        image = fluid.layers.data(
            name='image', shape=[1, 28, 28], dtype='float32')
        image.stop_gradient = False
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
                    print(
                        'eval iter={}, avg loss {}, acc_top1 {}, acc_top5 {}'.
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
        quant_train_prog_pact = quant_aware(
            main_prog,
            place,
            config,
            for_test=False,
            act_preprocess_func=pact,
            optimizer_func=get_optimizer,
            exe=exe)

        quant_eval_prog = quant_aware(val_prog, place, config, for_test=True)
        train(quant_train_prog_pact)
        quant_eval_prog, int8_prog = convert(
            quant_eval_prog, place, config, save_int8=True)
        top1_2, top5_2 = test(quant_eval_prog)
        # values before quantization and after quantization should be close
        print("before quantization: top1: {}, top5: {}".format(top1_1, top5_1))
        print("after quantization: top1: {}, top5: {}".format(top1_2, top5_2))


if __name__ == '__main__':
    unittest.main()
