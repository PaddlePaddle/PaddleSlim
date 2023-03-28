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
sys.path.append("../../")
import paddle
import tempfile
from paddle.vision.models import resnet18
from paddleslim.quant import SlimQuantConfig as QuantConfig
from paddleslim.quant import SlimQAT
from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
from paddle.quantization.quanters.abs_max import FakeQuanterWithAbsMaxObserverLayer
from paddle.nn.quant.format import LinearDequanter, LinearQuanter


def load_model_and_count_layer(model_path, layer_types):
    layer2count = dict([(_layer, 0) for _layer in layer_types])
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        [
            inference_program,
            feed_target_names,
            fetch_targets,
        ] = paddle.static.load_inference_model(model_path, exe)
    for _op in inference_program.global_block().ops:
        if _op.type in layer2count:
            layer2count[_op.type] += 1
    paddle.disable_static()
    return layer2count


class TestQuantAwareTraining(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")
        self.init_case()
        self.dummy_input = paddle.rand([1, 3, 224, 224])
        self.temp_dir = tempfile.TemporaryDirectory(dir="./")
        self.path = os.path.join(self.temp_dir.name, 'qat')

    def tearDown(self):
        self.temp_dir.cleanup()

    def extra_qconfig(self, q_config):
        """The subclass of TestQuantAwareTraining can implement the function to add more special configuration."""
        pass

    def init_case(self):
        quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
        self.quantizer_type = FakeQuanterWithAbsMaxObserverLayer
        self.quantizer_type_in_static = "fake_quantize_dequantize_moving_average_abs_max"
        self.q_config = QuantConfig(activation=None, weight=None)
        self.q_config.add_type_config(
            paddle.nn.Conv2D, activation=quanter, weight=quanter)
        self.extra_qconfig(self.q_config)

    def _count_layers(self, model, layer_type):
        count = 0
        for _layer in model.sublayers(True):
            if isinstance(_layer, layer_type):
                count += 1
        return count

    def test_quantize(self):
        model = resnet18()
        conv_count = self._count_layers(model, paddle.nn.Conv2D)
        qat = SlimQAT(self.q_config)
        quant_model = qat.quantize(model, inplace=True, inputs=self.dummy_input)
        quant_model.train()
        out = quant_model(self.dummy_input)
        out.backward()
        quantizer_cnt = self._count_layers(quant_model, self.quantizer_type)
        self.assertEqual(quantizer_cnt, conv_count * 2)

    def test_convert(self):
        model = resnet18()
        conv_count = self._count_layers(model, paddle.nn.Conv2D)
        qat = SlimQAT(self.q_config)
        quant_model = qat.quantize(model, inplace=True, inputs=self.dummy_input)
        converted_model = qat.convert(quant_model, inplace=True)

        # check count of LinearQuanter and LinearDequanter in dygraph
        quantizer_count_in_dygraph = self._count_layers(converted_model,
                                                        LinearQuanter)
        dequantizer_count_in_dygraph = self._count_layers(
            converted_model, LinearDequanter)

        self.assertEqual(quantizer_count_in_dygraph, conv_count)
        self.assertEqual(dequantizer_count_in_dygraph, conv_count * 2)

        # check count of LinearQuanter and LinearDequanter in static model saved by jit.save
        save_path = os.path.join(self.path, 'converted_model')
        quant_model.eval()
        paddle.jit.save(quant_model, save_path, input_spec=[self.dummy_input])

        layer2count = load_model_and_count_layer(
            save_path, ["quantize_linear", "dequantize_linear"])
        quantizer_count_in_static_model = layer2count['quantize_linear']
        dequantizer_count_in_static_model = layer2count['dequantize_linear']
        self.assertEqual(quantizer_count_in_dygraph,
                         quantizer_count_in_static_model)
        self.assertEqual(dequantizer_count_in_dygraph,
                         dequantizer_count_in_static_model)

    def test_trace_qat_graph(self):
        model = resnet18()
        qat = SlimQAT(self.q_config)
        quant_model = qat.quantize(model, inplace=True, inputs=self.dummy_input)
        quantizer_count_indygraph = self._count_layers(quant_model,
                                                       self.quantizer_type)
        save_path = os.path.join(self.path, 'qat_model')
        quant_model.eval()
        paddle.jit.save(quant_model, save_path, input_spec=[self.dummy_input])

        layer2count = load_model_and_count_layer(
            save_path, [self.quantizer_type_in_static])
        quantizer_count_in_static_model = layer2count[
            self.quantizer_type_in_static]
        self.assertEqual(quantizer_count_indygraph,
                         quantizer_count_in_static_model)


if __name__ == '__main__':
    unittest.main()
