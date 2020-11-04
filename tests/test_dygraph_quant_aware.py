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
sys.path.append("../")
import unittest
import logging
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.container import Sequential
from paddle.fluid.dygraph.nn import Conv2D
from paddle.fluid.dygraph.nn import Pool2D
from paddle.fluid.dygraph.nn import Linear
from paddle.fluid.log_helper import get_logger

from paddleslim.quant import quant_aware

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')

class ImperativeLenet(fluid.dygraph.Layer):
    def __init__(self, num_classes=10, classifier_activation='softmax'):
        super(ImperativeLenet, self).__init__()
        self.features = Sequential(
            Conv2D(
                num_channels=1,
                num_filters=6,
                filter_size=3,
                stride=1,
                padding=1),
            Pool2D(
                pool_size=2, pool_type='max', pool_stride=2),
            Conv2D(
                num_channels=6,
                num_filters=16,
                filter_size=5,
                stride=1,
                padding=0),
            Pool2D(
                pool_size=2, pool_type='max', pool_stride=2))

        self.fc = Sequential(
            Linear(
                input_dim=400,
                output_dim=120),
            Linear(
                input_dim=120,
                output_dim=84),
            Linear(
                input_dim=84,
                output_dim=num_classes,
                act=classifier_activation))

    def forward(self, inputs):
        x = self.features(inputs)

        x = fluid.layers.flatten(x, 1)
        x = self.fc(x)
        return x

class TestImperativeQat(unittest.TestCase):
    """
    QAT = quantization-aware training
    """

    def test_qat_acc(self):
        with fluid.dygraph.guard():
            lenet = ImperativeLenet()
            print("sublayers num: 1", len(lenet.sublayers()))
            quant_lenet = quant_aware(lenet)
            print("sublayers num: 2", len(quant_lenet.sublayers()))
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=32, drop_last=True)
            test_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=32)

            def train(model):
                adam = AdamOptimizer(
                learning_rate=0.001, parameter_list=model.parameters())
                epoch_num = 1
                for epoch in range(epoch_num):
                    model.train()
                    for batch_id, data in enumerate(train_reader()):
                        x_data = np.array([x[0].reshape(1, 28, 28)
                                        for x in data]).astype('float32')
                        y_data = np.array(
                            [x[1] for x in data]).astype('int64').reshape(-1, 1)

                        img = fluid.dygraph.to_variable(x_data)
                        label = fluid.dygraph.to_variable(y_data)
                        out = model(img)
                        acc = fluid.layers.accuracy(out, label)
                        loss = fluid.layers.cross_entropy(out, label)
                        avg_loss = fluid.layers.mean(loss)
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
                avg_acc = [[],[]]
                for batch_id, data in enumerate(test_reader()):
                    x_data = np.array([x[0].reshape(1, 28, 28)
                                       for x in data]).astype('float32')
                    y_data = np.array(
                        [x[1] for x in data]).astype('int64').reshape(-1, 1)

                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)

                    out = model(img)
                    acc_top1 = fluid.layers.accuracy(
                        input=out, label=label, k=1)
                    acc_top5 = fluid.layers.accuracy(
                        input=out, label=label, k=5)
                    avg_acc[0].append(acc_top1.numpy())
                    avg_acc[1].append(acc_top5.numpy())
                    if batch_id % 100 == 0:
                        _logger.info(
                            "Test | step {}: acc1 = {:}, acc5 = {:}".
                            format(batch_id,
                                   acc_top1.numpy(), acc_top5.numpy()))

                _logger.info(
                            "Test |Average: acc_top1 {}, acc_top5 {}".format(
                    np.mean(avg_acc[0]), np.mean(avg_acc[1])))  
                return np.mean(avg_acc[0]), np.mean(avg_acc[1])

            train(lenet)
            top1_1, top5_1 = test(lenet)

            quant_lenet.__init__()
            train(quant_lenet)
            top1_2, top5_2 = test(quant_lenet)

            # values before quantization and after quantization should be close
            _logger.info("Before quantization: top1: {}, top5: {}".format(top1_1, top5_1))
            _logger.info("After quantization: top1: {}, top5: {}".format(top1_2, top5_2))
if __name__ == '__main__':
    unittest.main()
