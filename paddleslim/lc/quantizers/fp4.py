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
from .base_quantizer import BaseQuantizer
from .quant_func import quantize_fp4, dequantize_fp4, double_quant, double_dequant


class FP4Quantizer(BaseQuantizer):

    def __init__(self, block_size=64, double_quant=False):
        super(FP4Quantizer, self).__init__()
        self.block_size = block_size
        self.double_quant = double_quant
        self.quant_state = "fp4_quant_state"

    def quantize(self, x: paddle.Tensor):
        out, quant_scale = quantize_fp4(x, self.block_size)
        if self.double_quant:
            qquant_scale, double_quant_scale, offset, code = double_quant(quant_scale, code=None, blocksize=self.block_size)
            self.state = [qquant_scale, double_quant_scale, offset, code]
        else:
            self.state = quant_scale
        return out

    def dequantize(self, x: paddle.Tensor):
        if self.double_quant:
            qquant_scale, double_quant_scale, offset, code = self.state
            quant_scale = double_dequant(qquant_scale, offset, code, double_quant_scale, self.block_size)
        else:
            quant_scale = self.state 
        
        return dequantize_fp4(
            x, quant_scale, blocksize=self.block_size).cast(x.dtype).reshape((x.shape[-1], -1))

    def matmul(self, x: paddle.Tensor, y: paddle.Tensor, bias: paddle.Tensor):
        return x @ self.dequantize(y) + bias
