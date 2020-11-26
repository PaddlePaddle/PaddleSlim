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
import paddle
from paddleslim.common import VarCollector
from static_case import StaticCase
sys.path.append("../demo")
from models import MobileNet
from layers import conv_bn_layer
import paddle.dataset.mnist as reader
import numpy as np


class TestAnalysisHelper(StaticCase):
    def test_analysis_helper(self):
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

        places = paddle.static.cuda_places() if paddle.is_compiled_with_cuda(
        ) else paddle.static.cpu_places()
        exe = paddle.static.Executor(places[0])

        def transform(x):
            return np.reshape(x, [1, 28, 28])

        train_dataset = paddle.vision.datasets.MNIST(
            mode='train', backend='cv2', transform=transform)
        train_loader = paddle.io.DataLoader(
            train_dataset,
            places=places,
            feed_list=[image, label],
            drop_last=True,
            return_list=False,
            batch_size=64)
        exe.run(paddle.static.default_startup_program())

        vars = ['conv2d_0.tmp_0', 'fc_0.tmp_0', 'fc_0.tmp_1', 'fc_0.tmp_2']
        var_collector1 = VarCollector(main_prog, vars, use_ema=True)
        values = var_collector1.abs_max_run(
            train_loader, exe, step=None, loss_name=avg_cost.name)
        vars = [v.name for v in main_prog.list_vars() if v.persistable]
        var_collector2 = VarCollector(main_prog, vars, use_ema=False)
        values = var_collector2.run(train_loader,
                                    exe,
                                    step=None,
                                    loss_name=avg_cost.name)
        var_collector2.pdf(values)


if __name__ == '__main__':
    unittest.main()
