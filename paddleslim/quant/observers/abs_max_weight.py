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
from .channel_wise import ChannelWiseObserver
from paddle.quantization.factory import ObserverFactory


class AbsMaxChannelWiseWeightObserver(ObserverFactory):
    r"""
    It collects channel-wise maximum absolute values of target weights.
    Args:
        bit_length(int, optional): Number of bits to represent an quantized integer in binary.
        dtype(str, optional): The data type of input tensor.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.
    Examples:
       .. code-block:: python
            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import AbsMaxChannelWiseWeightObserver
            quanter = AbsMaxChannelWiseWeightObserver()
            q_config = QuantConfig(activation=None, weight=quanter)
    """

    def __init__(self, quant_bits=8):
        super(AbsMaxChannelWiseWeightObserver, self).__init__(
            quant_bits=quant_bits)

    def _get_class(self):
        return AbsMaxChannelWiseWeightObserverLayer


class AbsMaxChannelWiseWeightObserverLayer(ChannelWiseObserver):
    def __init__(self, layer, quant_bits=8):
        super(AbsMaxChannelWiseWeightObserverLayer, self).__init__(
            layer,
            quant_bits=quant_bits,
            sign=True,
            symmetric=True, )
        self.quant_bits = quant_bits
        self.calibration_loss = float('inf')
        self.qmin, self.qmax = self.qmin_qmax
        self._layer = layer
        self._max = None
        self._scale = None
        self._zero_point = None

    def forward(self, inputs):
        if self._max is None:
            self._max = self._cal_abs_max(inputs)
        return inputs

    def _cal_abs_max(self, inputs):
        reduce_axis = tuple(
            [i for i in range(len(inputs.shape)) if i != self.quant_axis()])
        abs_max_values = paddle.max(paddle.abs(inputs), axis=reduce_axis)
        abs_max_values = paddle.where(
            abs_max_values == paddle.to_tensor(0, dtype=inputs.dtype),
            paddle.to_tensor(1e-8, dtype=inputs.dtype), abs_max_values)
        return abs_max_values

    def min_value(self) -> float:
        return 0.

    def max_value(self) -> float:
        return self._max

    def cal_thresholds(self):
        """ Compute thresholds for MAX function.
        """
        if self._scale is None:
            self._scale = self._max
        self._zero_point = paddle.zeros_like(self._scale)

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
