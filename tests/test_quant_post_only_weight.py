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
from paddleslim.quant import quant_post_only_weight
sys.path.append("../demo")
from models import MobileNet
from layers import conv_bn_layer
import paddle.dataset.mnist as reader
from paddle.fluid.framework import IrGraph
from paddle.fluid import core
import numpy as np


class TestQuantPostOnlyWeightCase1(unittest.TestCase):
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
        train_reader = paddle.io.batch(
            paddle.dataset.mnist.train(), batch_size=64)
        eval_reader = paddle.io.batch(
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

        def test(program, outputs=[avg_cost, acc_top1, acc_top5]):
            iter = 0
            result = [[], [], []]
            for data in train_reader():
                cost, top1, top5 = exe.run(program,
                                           feed=feeder.feed(data),
                                           fetch_list=outputs)
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
        top1_1, top5_1 = test(val_prog)
        fluid.io.save_inference_model(
            dirname='./test_quant_post',
            feeded_var_names=[image.name, label.name],
            target_vars=[avg_cost, acc_top1, acc_top5],
            main_program=val_prog,
            executor=exe,
            model_filename='model',
            params_filename='params')

        quant_post_only_weight(
            model_dir='./test_quant_post',
            save_model_dir='./test_quant_post_inference',
            model_filename='model',
            params_filename='params',
            generate_test_model=True)
        quant_post_prog, feed_target_names, fetch_targets = fluid.io.load_inference_model(
            dirname='./test_quant_post_inference/test_model', executor=exe)
        top1_2, top5_2 = test(quant_post_prog, fetch_targets)
        print("before quantization: top1: {}, top5: {}".format(top1_1, top5_1))
        print("after quantization: top1: {}, top5: {}".format(top1_2, top5_2))


if __name__ == '__main__':
    unittest.main()
