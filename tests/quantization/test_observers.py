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
sys.path.append("../../")
import os
import unittest
import paddle
import tempfile
from paddle.vision.models import resnet18
from paddle.quantization import QuantConfig
from paddle.quantization import PTQ

from paddleslim.quant.observers import HistObserver, KLObserver, EMDObserver, MSEObserver, AVGObserver
from paddleslim.quant.observers.hist import PercentHistObserverLayer
from paddleslim.quant.observers.kl import KLObserverLayer
from paddleslim.quant.observers.mse import MSEObserverLayer
from paddleslim.quant.observers.avg import AVGObserverLayer
from paddleslim.quant.observers.emd import EMDObserverLayer
from paddleslim.quant.observers.kl import KLObserverLayer
from paddle.nn.quant.format import LinearDequanter, LinearQuanter


class TestPTQObserver(unittest.TestCase):
    def __init__(self, observer, observer_type, *args, **kvargs):
        super(TestPTQObserver, self).__init__(*args, **kvargs)
        self.observer = observer
        self.observer_type = observer_type

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
            paddle.nn.Conv2D, activation=self.observer, weight=self.observer)

    def _count_layers(self, model, layer_type):
        count = 0
        for _layer in model.sublayers(True):
            if isinstance(_layer, layer_type):
                count += 1
        return count

    def test_quantize(self):
        model = resnet18()
        conv_count = self._count_layers(model, paddle.nn.Conv2D)
        ptq = PTQ(self.q_config)
        model.eval()
        quant_model = ptq.quantize(model, inplace=False)
        zero_input = paddle.zeros_like(self.dummy_input)
        out = quant_model(zero_input)
        out = quant_model(self.dummy_input)
        out = quant_model(zero_input)
        out = quant_model(self.dummy_input + 1.)
        quantizer_cnt = self._count_layers(quant_model, self.observer_type)
        self.assertEqual(quantizer_cnt, 2 * conv_count)

    def test_convert(self):
        model = resnet18()
        conv_count = self._count_layers(model, paddle.nn.Conv2D)
        ptq = PTQ(self.q_config)
        model.eval()
        quant_model = ptq.quantize(model, inplace=False)
        out = quant_model(self.dummy_input)
        converted_model = ptq.convert(quant_model, inplace=False)

        # check count of LinearQuanter and LinearDequanter in dygraph
        quantizer_count_in_dygraph = self._count_layers(converted_model,
                                                        LinearQuanter)
        dequantizer_count_in_dygraph = self._count_layers(
            converted_model, LinearDequanter)
        self.assertEqual(quantizer_count_in_dygraph, conv_count)
        self.assertEqual(dequantizer_count_in_dygraph, conv_count * 2)


observer_suite = unittest.TestSuite()
observer_suite.addTest(
    TestPTQObserver(
        observer=HistObserver(sign=True, symmetric=True),
        observer_type=PercentHistObserverLayer))
observer_suite.addTest(
    TestPTQObserver(
        observer=HistObserver(sign=False, symmetric=True),
        observer_type=PercentHistObserverLayer))
observer_suite.addTest(
    TestPTQObserver(
        observer=HistObserver(sign=True, symmetric=False),
        observer_type=PercentHistObserverLayer))
observer_suite.addTest(
    TestPTQObserver(
        observer=HistObserver(sign=False, symmetric=False),
        observer_type=PercentHistObserverLayer))
observer_suite.addTest(
    TestPTQObserver(
        observer=KLObserver(bins_count=256), observer_type=KLObserverLayer))

observer_suite.addTest(
    TestPTQObserver(observer=AVGObserver(), observer_type=AVGObserverLayer))
observer_suite.addTest(
    TestPTQObserver(observer=EMDObserver(), observer_type=EMDObserverLayer))
observer_suite.addTest(
    TestPTQObserver(observer=MSEObserver(), observer_type=MSEObserverLayer))
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(observer_suite)
