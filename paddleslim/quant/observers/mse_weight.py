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

import numpy as np
import paddle
from paddle.quantization.factory import ObserverFactory
from .abs_max_weight import AbsMaxChannelWiseWeightObserverLayer


class MSEChannelWiseWeightObserver(ObserverFactory):
    r"""
    It collects channel-wise maximum absolute values and calculates the quantization scales by minimizing
    the quantization MSE error.
    Args:
        bit_length(int, optional): Number of bits to represent an quantized integer in binary.
        dtype(str, optional): The data type of input tensor.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.
    Examples:
       .. code-block:: python
            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import MSEChannelWiseWeightObserver
            quanter = MSEChannelWiseWeightObserver()
            q_config = QuantConfig(activation=None, weight=quanter)
    """

    def __init__(self, quant_bits=8):
        super(MSEChannelWiseWeightObserver, self).__init__(
            quant_bits=quant_bits)

    def _get_class(self):
        return MSEChannelWiseWeightObserverLayer


class MSEChannelWiseWeightObserverLayer(AbsMaxChannelWiseWeightObserverLayer):
    def __init__(self, layer, quant_bits=8):
        super(MSEChannelWiseWeightObserverLayer, self).__init__(
            layer, quant_bits=quant_bits)

    def _cal_abs_max(self, inputs):
        reduce_axis = tuple(
            [i for i in range(len(inputs.shape)) if i != self.quant_axis()])
        abs_max_values = paddle.max(paddle.abs(inputs), axis=reduce_axis)
        abs_max_values = paddle.where(abs_max_values == np.float32(0.0),
                                      np.float32(1e-8), abs_max_values)
        minimum_loss = paddle.full(abs_max_values.shape, float('inf'))
        result = abs_max_values
        factor = 0.3
        while factor <= 1.0:
            scales = factor * abs_max_values
            factor += 0.02
            expand_scales = paddle.unsqueeze(scales, axis=reduce_axis)
            quant_var = paddle.clip(
                paddle.round(inputs / expand_scales * self.qmax), self.qmin,
                self.qmax)
            quant_dequant_var = quant_var / self.qmax * expand_scales

            mse_loss = ((inputs - quant_dequant_var)**2).mean(axis=reduce_axis)
            result = paddle.where(mse_loss < minimum_loss, scales, result)
            minimum_loss = paddle.minimum(mse_loss, minimum_loss)

        return result
