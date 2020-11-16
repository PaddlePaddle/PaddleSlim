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
import paddle
import unittest
import numpy as np
from static_case import StaticCase
from paddleslim.nas.darts import DARTSearch
from layers import conv_bn_layer


class TestDARTS(unittest.TestCase):
    def test_darts(self):
        class SuperNet(paddle.nn.Layer):
            def __init__(self):
                super(SuperNet, self).__init__()
                self._method = 'DARTS'
                self._steps = 1
                self.stem = paddle.nn.Conv2D(
                    in_channels=1, out_channels=3, kernel_size=3, padding=1)
                self.classifier = paddle.nn.Linear(
                    in_features=2352, out_features=10)
                self._multiplier = 4
                self._primitives = [
                    'none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect',
                    'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3',
                    'dil_conv_5x5'
                ]
                self._initialize_alphas()

            def _initialize_alphas(self):
                self.alphas_normal = self.create_parameter(
                    shape=[14, 8], dtype="float32")
                self.alphas_reduce = self.create_parameter(
                    shape=[14, 8], dtype="float32")
                self._arch_parameters = [
                    self.alphas_normal,
                    self.alphas_reduce,
                ]

            def arch_parameters(self):
                return self._arch_parameters

            def forward(self, input):
                out = self.stem(input) * self.alphas_normal[0][
                    0] * self.alphas_reduce[0][0]
                out = paddle.reshape(out, [0, -1])
                logits = self.classifier(out)
                return logits

            def _loss(self, input, label):
                logits = self.forward(input)
                return paddle.mean(
                    paddle.nn.functional.softmax_with_cross_entropy(logits,
                                                                    label))

        def batch_generator(reader):
            def wrapper():
                batch_data = []
                batch_label = []
                for sample in reader():
                    image = np.array(sample[0]).reshape(1, 28, 28)
                    label = np.array(sample[1]).reshape(1)
                    batch_data.append(image)
                    batch_label.append(label)
                    if len(batch_data) == 128:
                        batch_data = np.array(batch_data, dtype='float32')
                        batch_label = np.array(batch_label, dtype='int64')
                        yield [batch_data, batch_label]
                        batch_data = []
                        batch_label = []

            return wrapper

        place = paddle.CUDAPlace(0)
        model = SuperNet()
        trainset = paddle.dataset.mnist.train()
        validset = paddle.dataset.mnist.test()
        train_reader = batch_generator(trainset)
        valid_reader = batch_generator(validset)
        searcher = DARTSearch(
            model, train_reader, valid_reader, place, num_epochs=5)
        searcher.train()


if __name__ == '__main__':
    unittest.main()
