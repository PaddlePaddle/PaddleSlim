# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import paddle.nn as nn
from paddleslim.lc.quantizers import NF4Quantizer
from .linear import Linear4bit


class NF4Linear(Linear4bit):
    def __init__(
            self,
            linear: nn.Linear,
            block_size=64,
            use_double_quant=False, ):
        super(NF4Linear, self).__init__(linear, quant_type="nf4")
        self.block_size = block_size
        self.double_quant = use_double_quant
        self.quantizer = NF4Quantizer(block_size, self.double_quant)

    def quantize(self):
        quantized_weight = self.quantizer.quantize(self.linear.weight).reshape([self.out_features // 2, self.in_features])
        del self.linear.weight
        paddle.assign(quantized_weight, self.quant_weight)

    def forward(self, x):
        return self.quantizer.matmul(x, self.quant_weight.reshape([-1, 1]), self.linear.bias)
