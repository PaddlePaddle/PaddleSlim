# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import tempfile
sys.path.append("../../")
import paddle
from paddle.vision.models import resnet18
from paddleslim.core.graph_tracer import GraphTracer
from paddle.vision.models import resnet18


class TestConvBNConstraints(unittest.TestCase):
    def _count_layers(self, model, layer_type):
        count = 0
        for _layer in model.sublayers(True):
            if isinstance(_layer, layer_type):
                count += 1
        return count

    def test_conv_bn(self):
        paddle.set_device("cpu")
        model = resnet18()
        conv_count = self._count_layers(model, paddle.nn.Conv2D)
        x = paddle.rand([1, 3, 224, 224])
        tracer = GraphTracer(model)
        tracer(x)
        conv_bn_pairs = tracer.graph.find_conv_bn()
        self.assertEqual(len(conv_bn_pairs), conv_count)

        conv_names = set()
        bn_names = set()
        for _layer in model.sublayers():
            if isinstance(_layer, paddle.nn.Conv2D):
                conv_names.add(_layer.full_name())
            elif isinstance(_layer, paddle.nn.BatchNorm2D):
                bn_names.add(_layer.full_name())

        for _conv, _bn in conv_bn_pairs:
            self.assertTrue(_bn.layer_name in bn_names)
            self.assertTrue(_conv.layer_name in conv_names)


if __name__ == '__main__':
    unittest.main()
