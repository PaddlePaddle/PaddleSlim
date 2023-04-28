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

import logging
import paddle
from paddle.quantization.factory import ObserverFactory
from .uniform import UniformObserver
from ...common import get_logger

_logger = get_logger(
    __name__,
    logging.INFO,
    fmt='%(asctime)s-%(levelname)s: %(message)s', )

GAMMA = -0.1
ZETA = 1.1


class ReconstructActObserver(ObserverFactory):
    r"""
    Activation observer.
    Args:
        ptq_observer (ObserverFactory): It collects channel-wise maximum absolute values and calculates the quantization scales.
        batch_nums (int): Total number of minibatchs used to calibrate quantized variables. Default is 10.
        qdrop(bool, optional): Whether need the noise caused by activation quantization. More details can be found in https://openreview.net/pdf?id=ySQH0oDyp7. Default is False.
        drop_prob(float, optional): The dropout probability of activation quantization, and it is valid only if qdrop is True. Default is 0.5.
    """

    def __init__(
            self,
            ptq_observer,
            batch_nums=10,
            qdrop=False,
            drop_prob=0.5, ):
        super(ReconstructActObserver, self).__init__(
            ptq_observer=ptq_observer,
            batch_nums=batch_nums,
            qdrop=qdrop,
            drop_prob=drop_prob)

    def _get_class(self):
        return ReconstructActObserverLayer


class ReconstructActObserverLayer(UniformObserver):
    def __init__(self,
                 layer,
                 ptq_observer=None,
                 batch_nums=10,
                 qdrop=False,
                 drop_prob=0.5):
        super(ReconstructActObserverLayer, self).__init__()
        self._ptq_observer = ptq_observer._instance(layer)
        self._quant_bits = self._ptq_observer._quant_bits
        self._sign = self._ptq_observer._sign
        self._symmetric = self._ptq_observer._symmetric
        self._qmin, self._qmax = self.qmin_qmax

        self._batch_nums = batch_nums
        self._qdrop = qdrop
        self._drop_prob = drop_prob
        self._current_iters = 0

    def forward(self, inputs):
        """ Calculate forward pass.
        """
        self._current_iters += 1
        if self._current_iters <= self._batch_nums:
            return self._ptq_observer(inputs)

        if self._qdrop:
            quantized_inputs = paddle.round(self._quant(inputs, self.scales()))
            dequantized_inputs = self._dequant(quantized_inputs, self.scales())
            quant_noise = inputs - dequantized_inputs
            random_noise = paddle.nn.functional.dropout(
                quant_noise, p=self._drop_prob)
            return inputs - random_noise

        return inputs

    def set_batch_nums(self, batch_nums):
        self._batch_nums = batch_nums

    def cal_thresholds(self):
        """ Compute thresholds.
        """
        self._ptq_observer.cal_thresholds()

    def _quant(self, x, scale):
        s = scale / self._qmax
        quant_x = x / s
        return quant_x

    def _dequant(self, x, scale):
        s = scale / self._qmax
        dequant_x = s * x
        return dequant_x

    def min_value(self) -> float:
        """ The minimum value of floating-point numbers."""
        return self._ptq_observer._min

    def max_value(self) -> float:
        """ The maximum value of floating-point numbers."""
        return self._ptq_observer._max

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
        return self._ptq_observer.scales()

    def zero_points(self):
        """ Return output zero points.
        """
        return self._ptq_observer.zero_points()
