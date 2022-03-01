# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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
import os
sys.path.append(".")
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)

import paddle
import paddle.dataset.mnist as reader
import unittest
from paddleslim.source_free.auto_compression import AutoCompression
from paddleslim.source_free.strategy_config import *
from static_case import StaticCase
sys.path.append("../demo")
from models import MobileNet
from layers import conv_bn_layer
import numpy as np

paddle.enable_static()


class TestSourceFree(StaticCase):
    def setUp(self):
        image = paddle.static.data(
            name='image', shape=[None, 1, 28, 28], dtype='float32')
        label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
        model = MobileNet()
        out = model.net(input=image, class_dim=10)
        cost = paddle.nn.functional.loss.cross_entropy(input=out, label=label)
        avg_cost = paddle.mean(x=cost)
        acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
        acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)
        optimizer = paddle.optimizer.Momentum(
            momentum=0.9,
            learning_rate=0.01,
            weight_decay=paddle.regularizer.L2Decay(4e-5))
        optimizer.minimize(avg_cost)
        main_prog = paddle.static.default_main_program()
        val_prog = main_prog.clone(for_test=True)

        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())

        def transform(x):
            return np.reshape(x, [1, 28, 28])

        train_dataset = paddle.vision.datasets.MNIST(
            mode='train', backend='cv2', transform=transform)
        self.train_loader = paddle.io.DataLoader(
            train_dataset,
            places=place,
            feed_list=[image, label],
            drop_last=True,
            batch_size=64,
            return_list=False)

        test_dataset = paddle.vision.datasets.MNIST(
            mode='test', backend='cv2', transform=transform)
        self.valid_loader = paddle.io.DataLoader(
            test_dataset,
            places=place,
            feed_list=[image, label],
            batch_size=64,
            return_list=False)

        def train(program):
            iter = 0
            for data in self.train_loader():
                cost, top1, top5 = exe.run(
                    program,
                    feed=data,
                    fetch_list=[avg_cost, acc_top1, acc_top5])
                iter += 1
                if iter % 100 == 0:
                    print(
                        'train iter={}, avg loss {}, acc_top1 {}, acc_top5 {}'.
                        format(iter, cost, top1, top5))

        train(main_prog)
        top1_1 = self.eval_function(
            exe=exe,
            place=place,
            program=val_prog,
            test_feed_names=[image, label],
            test_fetch_list=[avg_cost, acc_top1, acc_top5])
        paddle.fluid.io.save_inference_model(
            dirname='./test_infer_model',
            feeded_var_names=[image.name, label.name],
            target_vars=[avg_cost, acc_top1, acc_top5],
            main_program=val_prog,
            executor=exe,
            model_filename='model',
            params_filename='params')

    def eval_function(self, exe, place, program, test_feed_names,
                      test_fetch_list):
        iter = 0
        result = [[], [], []]

        for data in self.valid_loader():
            cost, top1, top5 = exe.run(program,
                                       feed=data,
                                       fetch_list=test_fetch_list)
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

    def test_ptq_hpo(self):
        default_ptq_config = {
            "quantize_op_types": ["conv2d", "depthwise_conv2d", "mul"],
            "weight_bits": 8,
            "activation_bits": 8,
        }

        default_hpo_config = {
            "ptq_algo": ["KL", "hist"],
            "bias_correct": [True],
            "weight_quantize_type": ["channel_wise_abs_max"],
            "hist_percent": [0.9999, 0.99999],
            "batch_size": [4, 16],
            "batch_num": [4, 16],
            "max_quant_count": 2
        }
        ac = AutoCompression(
            model_dir='./test_infer_model/',
            model_filename='model',
            params_filename='params',
            save_dir='./test_ptq_hpo_output',
            strategy_config={
                "QuantizationConfig": QuantizationConfig(**default_ptq_config),
                "HyperParameterOptimizationConfig":
                HyperParameterOptimizationConfig(**default_hpo_config)
            },
            train_config=None,
            train_dataloader=self.train_loader,
            eval_callback=self.valid_loader)

        ac.compression()

    def test_qat_dis(self):
        default_ptq_config = {
            "quantize_op_types": ["conv2d", "depthwise_conv2d", "mul"],
            "weight_bits": 8,
            "activation_bits": 8,
            "is_full_quantize": False,
            "not_quant_pattern": ["skip_quant"],
        }

        default_distill_config = {
            "distill_loss": 'l2_loss',
            "distill_node_pair": ["teacher_linear_1.tmp_0", "linear_1.tmp_0"],
            "distill_lambda": 1.0,
            "teacher_model_dir": "./test_infer_model/",
            "teacher_model_filename": 'model',
            "teacher_params_filename": 'params',
        }

        default_train_config = {
            "epochs": 1,
            "optimizer": "SGD",
            "learning_rate": 0.0001,
            "eval_iter": 100,
            "origin_metric": 0.97,
        }

        ac = AutoCompression(
            model_dir='./test_infer_model/',
            model_filename='model',
            params_filename='params',
            save_dir='./test_qat_dis_output',
            strategy_config={
                "QuantizationConfig": QuantizationConfig(**default_ptq_config),
                "DistillationConfig":
                DistillationConfig(**default_distill_config)
            },
            train_config=TrainConfig(**default_train_config),
            train_dataloader=self.train_loader,
            eval_callback=self.eval_function)

        ac.compression()

    def test_prune_dis(self):

        default_prune_config = {'prune_algo': 'asp'}
        default_distill_config = {
            "distill_loss": 'l2_loss',
            "distill_node_pair": ["teacher_linear_1.tmp_0", "linear_1.tmp_0"],
            "distill_lambda": 1.0,
            "teacher_model_dir": "./test_infer_model/",
            "teacher_model_filename": 'model',
            "teacher_params_filename": 'params',
        }

        default_train_config = {
            "epochs": 1,
            "optimizer": "SGD",
            "learning_rate": 0.0001,
            "eval_iter": 100,
            "origin_metric": 0.97,
        }

        ac = AutoCompression(
            model_dir='./test_infer_model/',
            model_filename='model',
            params_filename='params',
            save_dir='./test_qat_dis_output',
            strategy_config={
                "PruneConfig": QuantizationConfig(**default_prune_config),
                "DistillationConfig":
                DistillationConfig(**default_distill_config)
            },
            train_config=TrainConfig(**default_train_config),
            train_dataloader=self.train_loader,
            eval_callback=self.eval_function)

        ac.compression()


if __name__ == '__main__':
    unittest.main()
