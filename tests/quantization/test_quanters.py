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
import paddle
import tempfile
from paddle.vision.models import resnet18
from paddle.quantization import QuantConfig
from paddle.quantization import QAT
from paddleslim.quant.quanters.lsq_act import ActLSQplusQuanterLayer, ActLSQplusQuanter
from paddleslim.quant.quanters.lsq_weight import WeightLSQplusQuanterLayer, WeightLSQplusQuanter
from paddle.nn.quant.format import LinearDequanter, LinearQuanter


class TestQATWithQuanters(unittest.TestCase):
    def __init__(self, act_observer, act_observer_type, weight_observer,
                 weight_observer_type, *args, **kvargs):
        super(TestQATWithQuanters, self).__init__(*args, **kvargs)
        self.act_observer = act_observer
        self.act_observer_type = act_observer_type
        self.weight_observer = weight_observer
        self.weight_observer_type = weight_observer_type

    def setUp(self):
        paddle.set_device("cpu")
        self.init_case()
        self.dummy_input = paddle.rand([1, 3, 224, 224])
        self.temp_dir = tempfile.TemporaryDirectory(dir="./")
        self.path = os.path.join(self.temp_dir.name, 'qat')

    def tearDown(self):
        self.temp_dir.cleanup()

    def runTest(self):
        self.test_quantize()
        self.test_convert()

    def init_case(self):
        self.q_config = QuantConfig(activation=None, weight=None)
        self.q_config.add_type_config(
            paddle.nn.Conv2D,
            activation=self.act_observer,
            weight=self.weight_observer)

    def _count_layers(self, model, layer_type):
        count = 0
        for _layer in model.sublayers(True):
            if isinstance(_layer, layer_type):
                count += 1
        return count

    def test_quantize(self):
        model = resnet18()
        conv_count = self._count_layers(model, paddle.nn.Conv2D)
        qat = QAT(self.q_config)
        model.train()
        quant_model = qat.quantize(model, inplace=False)
        out = quant_model(self.dummy_input)
        quantizer_cnt = self._count_layers(quant_model, self.act_observer_type)
        self.assertEqual(quantizer_cnt, conv_count)
        quantizer_cnt = self._count_layers(quant_model,
                                           self.weight_observer_type)
        self.assertEqual(quantizer_cnt, conv_count)

    def test_convert(self):
        model = resnet18()
        conv_count = self._count_layers(model, paddle.nn.Conv2D)
        qat = QAT(self.q_config)
        model.train()
        quant_model = qat.quantize(model, inplace=False)
        out = quant_model(self.dummy_input)
        converted_model = qat.convert(quant_model, inplace=False)

        # check count of LinearQuanter and LinearDequanter in dygraph
        quantizer_count_in_dygraph = self._count_layers(converted_model,
                                                        LinearQuanter)
        dequantizer_count_in_dygraph = self._count_layers(
            converted_model, LinearDequanter)
        self.assertEqual(quantizer_count_in_dygraph, conv_count)
        self.assertEqual(dequantizer_count_in_dygraph, conv_count * 2)


observer_suite = unittest.TestSuite()
observer_suite.addTest(
    TestQATWithQuanters(
        act_observer=ActLSQplusQuanter(),
        act_observer_type=ActLSQplusQuanterLayer,
        weight_observer=WeightLSQplusQuanter(),
        weight_observer_type=WeightLSQplusQuanterLayer))
observer_suite.addTest(
    TestQATWithQuanters(
        act_observer=ActLSQplusQuanter(),
        act_observer_type=ActLSQplusQuanterLayer,
        weight_observer=WeightLSQplusQuanter(per_channel=True),
        weight_observer_type=WeightLSQplusQuanterLayer))

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(observer_suite)
