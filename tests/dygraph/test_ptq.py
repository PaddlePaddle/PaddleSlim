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

import numpy as np
import sys
sys.path.append("../../")
import unittest
import logging
import paddle
import paddle.nn as nn
import paddle.fluid as fluid
from paddle.fluid.log_helper import get_logger
import paddle.vision.transforms as T

from paddleslim import PTQ

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class ImperativeLenet(nn.Layer):
    def __init__(self, num_classes=10):
        super(ImperativeLenet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2D(
                in_channels=1,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False),
            nn.BatchNorm2D(6),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=2, stride=2),
            nn.Conv2D(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0),
            nn.BatchNorm2D(16),
            nn.PReLU(),
            nn.MaxPool2D(
                kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(
                in_features=400, out_features=120),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=120, out_features=84),
            nn.Sigmoid(),
            nn.Linear(
                in_features=84, out_features=num_classes),
            nn.Softmax())

    def forward(self, inputs):
        x = self.features(inputs)

        x = fluid.layers.flatten(x, 1)
        x = self.fc(x)
        return x


class TestPTQ(unittest.TestCase):
    """
    Test dygraph post training quantization.
    """

    def calibrate(self, model, test_reader, batch_num=10):
        model.eval()
        for batch_id, data in enumerate(test_reader):
            img = paddle.to_tensor(data[0])
            img = paddle.reshape(img, [-1, 1, 28, 28])

            out = model(img)

            if batch_num + 1 >= batch_num:
                break

    def model_test(self, model, test_reader):
        model.eval()
        avg_acc = [[], []]
        for batch_id, data in enumerate(test_reader):
            img = paddle.to_tensor(data[0])
            img = paddle.reshape(img, [-1, 1, 28, 28])
            label = paddle.to_tensor(data[1])
            label = paddle.reshape(label, [-1, 1])

            out = model(img)

            acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)
            avg_acc[0].append(acc_top1.numpy())
            avg_acc[1].append(acc_top5.numpy())

            if batch_id % 100 == 0:
                _logger.info("Test | step {}: acc1 = {:}, acc5 = {:}".format(
                    batch_id, acc_top1.numpy(), acc_top5.numpy()))

        _logger.info("Test |Average: acc_top1 {}, acc_top5 {}".format(
            np.mean(avg_acc[0]), np.mean(avg_acc[1])))
        return np.mean(avg_acc[0]), np.mean(avg_acc[1])

    def model_train(self, model, train_reader):
        adam = paddle.optimizer.Adam(
            learning_rate=0.0001, parameters=model.parameters())
        epoch_num = 1
        for epoch in range(epoch_num):
            model.train()
            for batch_id, data in enumerate(train_reader):
                img = paddle.to_tensor(data[0])
                label = paddle.to_tensor(data[1])
                img = paddle.reshape(img, [-1, 1, 28, 28])
                label = paddle.reshape(label, [-1, 1])

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
                        format(epoch, batch_id, avg_loss.numpy(), acc.numpy()))

    def test_ptq(self):
        seed = 1
        np.random.seed(seed)
        paddle.static.default_main_program().random_seed = seed
        paddle.static.default_startup_program().random_seed = seed

        _logger.info("create the fp32 model")
        fp32_lenet = ImperativeLenet()

        _logger.info("prepare data")
        batch_size = 64
        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        train_dataset = paddle.vision.datasets.MNIST(
            mode='train', backend='cv2', transform=transform)
        val_dataset = paddle.vision.datasets.MNIST(
            mode='test', backend='cv2', transform=transform)

        place = paddle.CUDAPlace(0) \
            if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
        train_reader = paddle.io.DataLoader(
            train_dataset,
            drop_last=True,
            places=place,
            batch_size=batch_size,
            return_list=True)
        test_reader = paddle.io.DataLoader(
            val_dataset, places=place, batch_size=batch_size, return_list=True)

        _logger.info("train the fp32 model")
        self.model_train(fp32_lenet, train_reader)

        _logger.info("test fp32 model")
        fp32_top1, fp32_top5 = self.model_test(fp32_lenet, test_reader)

        _logger.info("quantize the fp32 model")
        quanter = PTQ()
        quant_lenet = quanter.quantize(fp32_lenet, fuse=True)

        _logger.info("calibrate")
        self.calibrate(quant_lenet, test_reader)

        _logger.info("save and test the quantized model")
        save_path = "./tmp/model"
        input_spec = paddle.static.InputSpec(
            shape=[None, 1, 28, 28], dtype='float32')
        quanter.save_quantized_model(
            quant_lenet, save_path, input_spec=[input_spec])
        quant_top1, quant_top5 = self.model_test(quant_lenet, test_reader)

        _logger.info("FP32 acc: top1: {}, top5: {}".format(fp32_top1,
                                                           fp32_top5))
        _logger.info("Int acc: top1: {}, top5: {}".format(quant_top1,
                                                          quant_top5))

        diff = 0.002
        self.assertTrue(
            fp32_top1 - quant_top1 < diff,
            msg="The acc of quant model is too lower than fp32 model")


if __name__ == '__main__':
    unittest.main()
