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

import numpy as np
import sys
sys.path.append("../")
import unittest
import logging
import types
import paddle
import paddle.nn as nn
from paddle.fluid.dygraph.nn import Conv2D
from paddle.fluid.dygraph.nn import Linear

from paddle.fluid.log_helper import get_logger

from paddleslim.quant import quant_aware

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class PACT(nn.Layer):
    def __init__(self, init_value=20):
        super(PACT, self).__init__()
        alpha_attr = paddle.ParamAttr(
            name=self.full_name() + ".pact",
            initializer=paddle.nn.initializer.Constant(value=init_value))
        self.alpha = self.create_parameter(
            shape=[1], attr=alpha_attr, dtype='float32')

    def forward(self, x):
        out_left = paddle.nn.functional.relu(x - self.alpha)
        out_right = paddle.nn.functional.relu(-self.alpha - x)
        x = x - out_left + out_right
        return x


class ImperativeLenet(nn.Layer):
    def __init__(self, num_classes=10, classifier_activation='softmax'):
        super(ImperativeLenet, self).__init__()
        self.features = paddle.nn.Sequential(
            Conv2D(
                num_channels=1,
                num_filters=6,
                filter_size=3,
                stride=1,
                padding=1),
            paddle.nn.Pool2D(
                pool_size=2, pool_type='max', pool_stride=2),
            Conv2D(
                num_channels=6,
                num_filters=16,
                filter_size=5,
                stride=1,
                padding=0),
            paddle.nn.Pool2D(
                pool_size=2, pool_type='max', pool_stride=2))

        self.fc = paddle.nn.Sequential(
            Linear(
                input_dim=400, output_dim=120),
            Linear(
                input_dim=120, output_dim=84),
            Linear(
                input_dim=84, output_dim=num_classes,
                act=classifier_activation))

    def forward(self, inputs):
        x = self.features(inputs)

        x = paddle.flatten(x, 1)
        x = self.fc(x)
        return x


class TestImperativeQat(unittest.TestCase):
    """
    QAT = quantization-aware training
    """

    def test_qat_acc(self):
        def train(model):
            adam = paddle.optimizer.Adam(
                learning_rate=0.0001, parameters=model.parameters())
            epoch_num = 1
            for epoch in range(epoch_num):
                model.train()
                for batch_id, data in enumerate(train_reader()):
                    img = paddle.to_tensor(data[0])
                    label = paddle.to_tensor(data[1])
                    out = model(img)
                    acc = paddle.metric.accuracy(out, label)
                    loss = paddle.nn.functional.loss.cross_entropy(out, label)
                    avg_loss = paddle.mean(loss)
                    avg_loss.backward()
                    adam.minimize(avg_loss)
                    model.clear_gradients()
                    if batch_id % 100 == 0:
                        _logger.info(
                            "Train | At epoch {} step {}: loss = {:}, acc= {:}".
                            format(epoch, batch_id,
                                   avg_loss.numpy(), acc.numpy()))

        def test(model):
            model.eval()
            avg_acc = [[], []]
            for batch_id, data in enumerate(test_reader()):
                img = paddle.to_tensor(data[0])
                label = paddle.to_tensor(data[1])

                out = model(img)
                acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
                acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)
                avg_acc[0].append(acc_top1.numpy())
                avg_acc[1].append(acc_top5.numpy())
                if batch_id % 100 == 0:
                    _logger.info(
                        "Test | step {}: acc1 = {:}, acc5 = {:}".format(
                            batch_id, acc_top1.numpy(), acc_top5.numpy()))

            _logger.info("Test |Average: acc_top1 {}, acc_top5 {}".format(
                np.mean(avg_acc[0]), np.mean(avg_acc[1])))
            return np.mean(avg_acc[0]), np.mean(avg_acc[1])

        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.static.CPUPlace()
        lenet = ImperativeLenet()
        quant_lenet = quant_aware(lenet, act_preprocess_layer=PACT)

        def transform(x):
            return np.reshape(np.array(x).astype('float32'), [1, 28, 28])

        train_dataset = paddle.vision.datasets.MNIST(
            mode='train', transform=transform)
        test_dataset = paddle.vision.datasets.MNIST(
            mode='test', transform=transform)
        train_reader = paddle.io.DataLoader(
            train_dataset,
            places=place,
            shuffle=True,
            drop_last=True,
            batch_size=32)
        test_reader = paddle.io.DataLoader(
            test_dataset, places=place, batch_size=32)

        print(paddle.summary(quant_lenet, (1, 1, 28, 28)))
        train(quant_lenet)
        top1_1, top5_1 = test(quant_lenet)

        lenet.__init__()
        print(paddle.summary(lenet, (1, 1, 28, 28)))
        train(lenet)
        top1_2, top5_2 = test(quant_lenet)

        # values before quantization and after quantization should be close
        _logger.info("Before quantization: top1: {}, top5: {}".format(top1_2,
                                                                      top5_2))
        _logger.info("After quantization: top1: {}, top5: {}".format(top1_1,
                                                                     top5_1))


if __name__ == '__main__':
    unittest.main()
