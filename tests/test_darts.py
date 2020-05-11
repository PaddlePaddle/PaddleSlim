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
import paddle
import unittest
import paddle.fluid as fluid
import numpy as np
from paddleslim.nas.darts import DARTSearch
from layers import conv_bn_layer


class TestDARTS(unittest.TestCase):
    def test_darts(self):
        class SuperNet(fluid.dygraph.Layer):
            def __init__(self):
                super(SuperNet, self).__init__()
                self._method = 'DARTS'
                self._steps = 1
                self.stem = fluid.dygraph.nn.Conv2D(
                    num_channels=1, num_filters=3, filter_size=3, padding=1)
                self.classifier = fluid.dygraph.nn.Linear(
                    input_dim=3072, output_dim=10)
                self._multiplier = 4
                self._primitives = [
                    'none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect',
                    'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3',
                    'dil_conv_5x5'
                ]
                self._initialize_alphas()

            def _initialize_alphas(self):
                self.alphas_normal = fluid.layers.create_parameter(
                    shape=[14, 8], dtype="float32")
                self.alphas_reduce = fluid.layers.create_parameter(
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
                out = fluid.layers.reshape(out, [0, -1])
                logits = self.classifier(out)
                return logits

            def _loss(self, input, label):
                logits = self.forward(input)
                return fluid.layers.reduce_mean(
                    fluid.layers.softmax_with_cross_entropy(logits, label))

        def batch_generator_creator():
            def __reader__():
                for _ in range(100):
                    batch_image = np.random.random(
                        size=[64, 1, 32, 32]).astype('float32')
                    batch_label = np.random.random(
                        size=[64, 1]).astype('int64')
                    yield batch_image, batch_label

            return __reader__

        place = fluid.CUDAPlace(0)
        with fluid.dygraph.guard(place):
            model = SuperNet()
            train_reader = batch_generator_creator()
            valid_reader = batch_generator_creator()
            searcher = DARTSearch(
                model, train_reader, valid_reader, place, num_epochs=1)
            searcher.train()


if __name__ == '__main__':
    unittest.main()
