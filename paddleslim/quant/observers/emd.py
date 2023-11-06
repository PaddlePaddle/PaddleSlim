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
from .uniform import UniformObserver
from paddle.quantization.factory import ObserverFactory


class EMDObserver(ObserverFactory):
    r"""
    It collects maximum absolute values of target tensor.
    Args:
        bit_length(int, optional): Number of bits to represent an quantized integer in binary.
        dtype(str, optional): The data type of input tensor.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.
    Examples:
       .. code-block:: python
            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
            quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.99)
            q_config = QuantConfig(activation=quanter, weight=quanter)
    """

    def __init__(self, quant_bits=8):
        super(EMDObserver, self).__init__(quant_bits=quant_bits)

    def _get_class(self):
        return EMDObserverLayer


class EMDObserverLayer(UniformObserver):
    def __init__(self, layer, quant_bits=8):
        super(EMDObserverLayer, self).__init__(quant_bits=quant_bits)
        self._quant_bits = quant_bits
        self._calibration_loss = float('inf')
        self.qmin, self.qmax = self.qmin_qmax

    def forward(self, inputs):
        """ Calculate forward pass.
        """
        self._scale = None
        self._zero_point = None
        self._min = None
        self._max = None
        self._emd_min, self._emd_max = self.cal_min_max(inputs)

        return inputs

    def cal_min_max(self, inputs):
        abs_max_value = float(paddle.max(paddle.flatten(inputs)))
        abs_max_value = 1e-8 if abs_max_value == 0.0 else abs_max_value
        s = 0.3
        scale_emd = abs_max_value
        while s <= 1.0:
            scale = s * abs_max_value
            s += 0.02
            bins = 2**(self._quant_bits - 1) - 1
            quant_var = paddle.clip(
                paddle.round(inputs / scale * self.qmax), -self.qmax - 1,
                self.qmax)
            quant_dequant_var = quant_var / self.qmax * scale

            emd_loss = paddle.abs(
                paddle.mean(inputs) - paddle.mean(quant_dequant_var)
            ) + paddle.abs(paddle.std(inputs) - paddle.std(quant_dequant_var))
            emd_loss = float(emd_loss)
            if emd_loss <= self._calibration_loss:
                self._calibration_loss = emd_loss
                scale_emd = scale
        return 0, scale_emd

    def cal_thresholds(self):
        """ Compute thresholds for MAX function.
        """
        if self._scale is not None:
            self._zero_point = 0
            return
        self._min, self._max = self._emd_min, self._emd_max
        self._scale, self._zero_point = self.cal_scales_zero_points()

    def min_value(self) -> float:
        return self._min

    def max_value(self) -> float:
        return self._max

    def bit_length(self):
        """ Return the bit length of quantized data.
        """
        return self._quant_bits

    def quant_axis(self):
        """ Return quantization axis.
        """
        return -1

    def scales(self):
        """ Return output scales.
        """
        if self._scale is None:
            self.cal_thresholds()
        return self._scale

    def zero_points(self):
        """ Return output zero points.
        """
        if self._zero_point is None:
            self.cal_thresholds()
        return self._zero_point
