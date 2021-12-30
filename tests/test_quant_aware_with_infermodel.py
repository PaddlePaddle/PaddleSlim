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
sys.path.append("../")
sys.path.append(".")
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)
import unittest
import paddle
from paddleslim.quant import quant_aware, convert
from paddleslim.quant import quant_aware_with_infermodel, export_quant_infermodel
from static_case import StaticCase
sys.path.append("../demo")
from models import MobileNet
from layers import conv_bn_layer
import paddle.dataset.mnist as reader
from paddle.fluid.framework import IrGraph
from paddle.fluid import core
import numpy as np


class TestQuantAwareWithInferModelCase1(StaticCase):
    def test_accuracy(self):
        float_infer_model_path_prefix = "./mv1_float_inference"

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

        #place = paddle.CPUPlace()
        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())

        def transform(x):
            return np.reshape(x, [1, 28, 28])

        train_dataset = paddle.vision.datasets.MNIST(
            mode='train', backend='cv2', transform=transform)
        test_dataset = paddle.vision.datasets.MNIST(
            mode='test', backend='cv2', transform=transform)

        train_loader = paddle.io.DataLoader(
            train_dataset,
            places=place,
            feed_list=[image, label],
            drop_last=True,
            batch_size=64,
            return_list=False)

        valid_loader = paddle.io.DataLoader(
            test_dataset,
            places=place,
            feed_list=[image, label],
            batch_size=64,
            return_list=False)

        def sample_generator_creator():
            def __reader__():
                for data in test_dataset:
                    image, label = data
                    yield image, label

            return __reader__

        def train(program):
            iter = 0
            for data in train_loader():
                cost, top1, top5 = exe.run(
                    program,
                    feed=data,
                    fetch_list=[avg_cost, acc_top1, acc_top5])
                iter += 1
                if iter % 100 == 0:
                    print(
                        'train iter={}, avg loss {}, acc_top1 {}, acc_top5 {}'.
                        format(iter, cost, top1, top5))

        def test(program, outputs=[avg_cost, acc_top1, acc_top5]):
            iter = 0
            result = [[], [], []]
            for data in valid_loader():
                cost, top1, top5 = exe.run(program,
                                           feed=data,
                                           fetch_list=outputs)
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
        top1_1, top5_1 = test(val_prog)
        paddle.static.save_inference_model(
            path_prefix=float_infer_model_path_prefix,
            feed_vars=[image, label],
            fetch_vars=[avg_cost, acc_top1, acc_top5],
            executor=exe,
            program=val_prog)

        quant_config = {
            'weight_quantize_type': 'channel_wise_abs_max',
            'activation_quantize_type': 'moving_average_abs_max',
            'not_quant_pattern': ['skip_quant'],
            'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul']
        }
        train_config = {
            "num_epoch": 1,  # training epoch num
            "max_iter": 20,
            "save_iter_step": 10,
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "use_pact": False,
            "quant_model_ckpt_path":
            "./quantaware_with_infermodel_checkpoints/",
            "teacher_model_path_prefix": float_infer_model_path_prefix,
            "model_path_prefix": float_infer_model_path_prefix,
            "distill_node_pair": [
                "teacher_fc_0.tmp_0", "fc_0.tmp_0",
                "teacher_batch_norm_24.tmp_4", "batch_norm_24.tmp_4",
                "teacher_batch_norm_22.tmp_4", "batch_norm_22.tmp_4",
                "teacher_batch_norm_18.tmp_4", "batch_norm_18.tmp_4",
                "teacher_batch_norm_13.tmp_4", "batch_norm_13.tmp_4",
                "teacher_batch_norm_5.tmp_4", "batch_norm_5.tmp_4"
            ]
        }

        def test_callback(compiled_test_program, feed_names, fetch_list,
                          checkpoint_name):
            outputs = fetch_list
            iter = 0
            result = [[], [], []]
            for data in valid_loader():
                cost, top1, top5 = exe.run(compiled_test_program,
                                           feed=data,
                                           fetch_list=fetch_list)
                iter += 1
                if iter % 100 == 0:
                    print('eval iter={}, avg loss {}, acc_top1 {}, acc_top5 {}'.
                          format(iter, cost, top1, top5))
                result[0].append(cost)
                result[1].append(top1)
                result[2].append(top5)
            print("quant model checkpoint: " + checkpoint_name +
                  ' avg loss {}, acc_top1 {}, acc_top5 {}'.format(
                      np.mean(result[0]),
                      np.mean(result[1]), np.mean(result[2])))
            return np.mean(result[1]), np.mean(result[2])

        def test_quant_aware_with_infermodel(exe, place):
            quant_aware_with_infermodel(
                exe,
                place,
                scope=None,
                train_reader=train_loader,
                quant_config=quant_config,
                train_config=train_config,
                test_callback=test_callback)

        def test_export_quant_infermodel(exe, place, checkpoint_path,
                                         quant_infermodel_save_path):
            export_quant_infermodel(
                exe,
                place,
                scope=None,
                quant_config=quant_config,
                train_config=train_config,
                checkpoint_path=checkpoint_path,
                export_inference_model_path_prefix=quant_infermodel_save_path)

        #place = paddle.CPUPlace()
        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        test_quant_aware_with_infermodel(exe, place)
        checkpoint_path = "./quantaware_with_infermodel_checkpoints/epoch_0_iter_10"
        quant_infermodel_save_path = "./quantaware_with_infermodel_export"
        test_export_quant_infermodel(exe, place, checkpoint_path,
                                     quant_infermodel_save_path)
        train_config["use_pact"] = True
        test_quant_aware_with_infermodel(exe, place)
        train_config["use_pact"] = False
        checkpoint_path = "./quantaware_with_infermodel_checkpoints/epoch_0_iter_10"
        quant_infermodel_save_path = "./quantaware_with_infermodel_export"
        test_export_quant_infermodel(exe, place, checkpoint_path,
                                     quant_infermodel_save_path)


if __name__ == '__main__':
    unittest.main()
