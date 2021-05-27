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
import logging
import numpy as np
import paddle
import paddle.fluid as fluid
from static_case import StaticCase
import paddle.dataset.mnist as reader
from paddleslim.models.dygraph import MobileNetV1
from paddleslim.dist import DML
from paddleslim.common import get_logger
logger = get_logger(__name__, level=logging.INFO)


class Model(paddle.nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = paddle.nn.Conv2D(
            in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool2d_avg = paddle.nn.AdaptiveAvgPool2D([1, 1])
        self.out = paddle.nn.Linear(256, 10)

    def forward(self, inputs):
        inputs = paddle.reshape(inputs, shape=[0, 1, 28, 28])
        y = self.conv(inputs)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, shape=[-1, 256])
        y = self.out(y)
        return y


class TestDML(unittest.TestCase):
    def test_dml(self):
        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()

        def transform(x):
            return np.reshape(x, [1, 28, 28])

        train_dataset = paddle.vision.datasets.MNIST(
            mode='train', backend='cv2', transform=transform)
        train_loader = paddle.io.DataLoader(
            train_dataset, places=place, drop_last=True, batch_size=64)

        models = [Model(), Model()]
        optimizers = []
        for cur_model in models:
            opt = paddle.optimizer.Momentum(
                0.1, 0.9, parameters=cur_model.parameters())
            optimizers.append(opt)
        dml_model = DML(models)
        dml_optimizer = dml_model.opt(optimizers)

        def train(train_loader, dml_model, dml_optimizer):
            dml_model.train()
            for step_id, (images, labels) in enumerate(train_loader):
                images, labels = paddle.to_tensor(images), paddle.to_tensor(
                    labels)
                labels = paddle.reshape(labels, [0, 1])

                logits = dml_model.forward(images)
                precs = [
                    paddle.metric.accuracy(
                        input=l, label=labels, k=1).numpy() for l in logits
                ]
                losses = dml_model.loss(logits, labels)
                dml_optimizer.minimize(losses)
                if step_id % 10 == 0:
                    print(step_id, precs)

        for epoch_id in range(10):
            train(train_loader, dml_model, dml_optimizer)


if __name__ == '__main__':
    unittest.main()
