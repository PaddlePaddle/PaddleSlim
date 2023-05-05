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
import numpy as np
import paddle
from paddle.utils import unique_name
from paddle.nn.initializer import Assign
from paddle.quantization.factory import ObserverFactory
from .uniform import UniformObserver
from ...common import get_logger

_logger = get_logger(
    __name__,
    logging.INFO,
    fmt='%(asctime)s-%(levelname)s: %(message)s', )

GAMMA = -0.1
ZETA = 1.1


class ReconstructWeightObserver(ObserverFactory):
    """
    Weight observer, used to optimize the rounding policy by reconstructing the intermediate output.
    Args:
        ptq_observer (ObserverFactory): It collects channel-wise maximum absolute values and calculates the quantization scales.
        batch_nums (int, ): Total number of minibatchs used to calibrate quantized variables. Default is 10.
    """

    def __init__(self, ptq_observer, batch_nums=10):
        super(ReconstructWeightObserver, self).__init__(
            ptq_observer=ptq_observer, batch_nums=batch_nums)

    def _get_class(self):
        return ReconstructWeightObserverLayer


class ReconstructWeightObserverLayer(UniformObserver):
    def __init__(self, layer, ptq_observer, batch_nums=10):
        super(ReconstructWeightObserverLayer, self).__init__()
        self._ptq_observer = ptq_observer._instance(layer)
        self._quant_bits = self._ptq_observer._quant_bits
        self._sign = self._ptq_observer._sign
        self._symmetric = self._ptq_observer._symmetric
        self._qmin, self._qmax = self.qmin_qmax

        self._batch_nums = batch_nums
        self._current_iters = 0
        self.alpha = None
        self._alpha_prefix = ("{}.round_alpha".format(layer.full_name()))

    def _init_alpha(self, weight, scale):
        """ Initialize alpha
        """
        quantized_weight = np.clip(
            self._quant(weight.numpy(), scale.numpy()), self._qmin, self._qmax)
        floor_weight = np.floor(quantized_weight)
        mantissa = quantized_weight - floor_weight
        init_alpha = -np.log((ZETA - GAMMA) / (mantissa - GAMMA) - 1)

        self._alpha_name = unique_name.generate(self._alpha_prefix)
        alpha_attr = paddle.ParamAttr(
            name=self._alpha_name,
            initializer=Assign(value=init_alpha),
            trainable=True)
        self.alpha = self.create_parameter(
            shape=weight.shape, attr=alpha_attr, dtype=weight.dtype)

    def forward(self, weights):
        """ Calculate forward pass.
        """
        self._current_iters += 1
        if self._current_iters < self._batch_nums:
            return self._ptq_observer(weights)

        if self._current_iters == self._batch_nums:
            weights = self._ptq_observer(weights)
            self._prepare_scale(self._ptq_observer.scales(), weights.shape)
            self._init_alpha(weights, self.scale)
            return weights

        h_alpha = self.compute_soft_rounding()
        quantized_weight = self._quant(weights, self.scale)
        floor_weight = (paddle.floor(quantized_weight) -
                        quantized_weight).detach() + quantized_weight
        clip_weight = paddle.clip(floor_weight + h_alpha, self._qmin,
                                  self._qmax)
        dequant_weight = self._dequant(clip_weight, self.scale)
        return dequant_weight

    def _prepare_scale(self, scale, weight_shape):
        if scale.shape[0] == 1:
            self.scale = scale
        else:
            self.scale = scale.reshape([scale.shape[0], 1])
            if len(weight_shape) == 2:
                self.scale = self.scale.repeat_interleave(
                    weight_shape[0], axis=1).t()
            else:
                self.scale = self.scale.repeat_interleave(
                    weight_shape[1] * weight_shape[2] * weight_shape[3], axis=1)
                self.scale = self.scale.reshape(weight_shape)

    def compute_soft_rounding(self):
        return paddle.clip(
            paddle.nn.functional.sigmoid(self.alpha) * (ZETA - GAMMA) + GAMMA,
            0, 1)

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
        return self._ptq_observer.quant_axis()

    def scales(self):
        """ Return output scales.
        """
        return self._ptq_observer.scales()

    def zero_points(self):
        """ Return output zero points.
        """
        return self._ptq_observer.zero_points()
