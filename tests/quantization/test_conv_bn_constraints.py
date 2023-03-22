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
from paddleslim.quant import SlimQuantConfig as QuantConfig
from paddleslim.quant import SlimQAT
from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
from paddleslim.quant.nn.conv_bn import QuantedConv2DBatchNorm
from paddleslim.quant.constraints import FreezedConvBNConstraint
from test_qat import TestQuantAwareTraining, load_model_and_count_layer


class TestConvBNConstraintsBaseCase(TestQuantAwareTraining):
    """ Common cases for testing 'quantize', 'convert' and 'jit.save' function."""

    def extra_qconfig(self, qconfig):
        qconfig.add_constraints(FreezedConvBNConstraint(freeze_bn_delay=1))


class TestConvBNConstraints(unittest.TestCase):
    """ More special cases on convolution and batch norm constraints."""

    def setUp(self):
        paddle.set_device("cpu")
        self.temp_dir = tempfile.TemporaryDirectory(dir="./")
        self.path = os.path.join(self.temp_dir.name, 'conv_bn_constraints')

    def tearDown(self):
        self.temp_dir.cleanup()

    def _count_layers(self, model, layer_type):
        count = 0
        for _layer in model.sublayers(True):
            if isinstance(_layer, layer_type):
                count += 1
        return count

    def _get_one_layer(self, model, layer_type):
        for _layer in model.sublayers(True):
            if isinstance(_layer, layer_type):
                return _layer
        return None

    def test_conv_bn(self):
        model = resnet18()
        conv_count = self._count_layers(model, paddle.nn.Conv2D)
        quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
        q_config = QuantConfig(activation=quanter, weight=quanter)
        # It will freeze the batch normaliztion after 'freeze_bn_delay' steps
        q_config.add_constraints(FreezedConvBNConstraint(freeze_bn_delay=1))

        qat = SlimQAT(q_config)
        x = paddle.rand([1, 3, 224, 224])
        paddle.jit.save(model, "./test_model", input_spec=[x])
        quant_model = qat.quantize(model, inplace=True, inputs=x)

        # check freeze_bn_delay
        qat_conv_bn_layer = self._get_one_layer(quant_model,
                                                QuantedConv2DBatchNorm)
        self.assertIsNotNone(qat_conv_bn_layer)
        self.assertFalse(qat_conv_bn_layer._freeze_bn)
        quant_model.train()
        out = quant_model(x)
        out.backward()
        out = quant_model(x)
        out.backward()
        self.assertTrue(qat_conv_bn_layer._freeze_bn)

        # check the count of QAT layers in QAT model
        qat_layer_count = self._count_layers(quant_model,
                                             QuantedConv2DBatchNorm)
        self.assertEqual(qat_layer_count, conv_count)

        # check the count of convolution and batch norm in saved static graph
        quant_model.eval()
        infer_model = qat.convert(quant_model, inplace=True)
        save_path = os.path.join(self.path, 'infer_model')
        paddle.jit.save(infer_model, save_path, input_spec=[x])
        layer2count = load_model_and_count_layer(save_path,
                                                 ['conv2d', 'batch_norm'])
        self.assertEqual(layer2count['conv2d'], conv_count)
        self.assertEqual(layer2count['batch_norm'], 0)


if __name__ == '__main__':
    unittest.main()
