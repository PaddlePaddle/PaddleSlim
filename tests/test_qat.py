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

import sys
import unittest
import numpy as np

sys.path.append("../")

import paddle
from paddle.vision.models import resnet18
from paddle.quantization import QuantConfig
from paddleslim.quant import SlimQAT
from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
from paddleslim.quant.nn.conv_bn import QuantedConv2DBatchNorm
from paddleslim.quant.constraints import FreezedConvBNConstraint


class TestQuantAwareTraining(unittest.TestCase):
    def test_quantize(self):
        model = resnet18()
        quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
        q_config = QuantConfig(activation=quanter, weight=quanter)
        # It will freeze the batch normaliztion after 'freeze_bn_delay' steps
        q_config.add_constraints(FreezedConvBNConstraint(freeze_bn_delay=1))
        qat = SlimQAT(q_config)
        x = paddle.rand([1, 3, 224, 224])
        quant_model = qat.quantize(model, inplace=True, inputs=x)
        quant_model.train()
        out = quant_model(x)
        out.backward()
        out = quant_model(x)
        out.backward()
        quant_model.eval()
        out = quant_model(x)
        print("------------------convert-----------------")
        infer_model = qat.convert(quant_model, inplace=True)
        infer_model(x)
        print(infer_model)
        paddle.jit.save(infer_model, "./infer_model", input_spec=[x])

        fuse_conv_bn_cnt = 0
        expected_quant_layer_cnt = 12
        for layer in quant_model.sublayers():
            if isinstance(layer, QuantedConv2DBatchNorm):
                fuse_conv_bn_cnt += 1
        self.assertEqual(fuse_conv_bn_cnt, expected_quant_layer_cnt)


if __name__ == '__main__':
    unittest.main()
