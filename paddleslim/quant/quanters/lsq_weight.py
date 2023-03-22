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

import abc
import paddle
import numpy as np
import math
from paddle.framework import ParamAttr
from paddle.nn import Layer
from paddle.nn.initializer import Constant
from paddle.utils import unique_name
from paddle.quantization.factory import QuanterFactory
from .lsq_func import LsqFunc, round
from .base_fake_quanter import BaseFakeQuanterLayer


class WeightLSQplusQuanter(QuanterFactory):
    r"""
    Weight quantizer. More details can be found in 
    https://arxiv.org/pdf/1902.08153.pdf and https://arxiv.org/pdf/2004.09576.pdf.
    Args:
        per_channel(bool): Whether layer-wise or channel-wise quantization, where True for layer-wise quantization and False for channel-wise quantization.
        batch_init(int): Number of batches that collect Gaussian approximation for the weight distribution in each layer.
        quant_linear(bool): whether the weight is from Linear.
        dtype(str): Trainable data type.
        name(str): The name of the layer.
        reduce_type(str): The reduce type which is needed when parallel training.
    Examples:
       .. code-block:: python
            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import ActLSQplusQuanter, WeightLSQplusQuanter
            weight_quanter = WeightLSQplusQuanter()
            act_quanter = ActLSQplusQuanter()
            q_config = QuantConfig(activation=act_quanter, weight=weight_quanter)
    """

    def __init__(self,
                 quant_bits=8,
                 sign=True,
                 symmetric=True,
                 per_channel=False,
                 batch_init=20,
                 quant_linear=False,
                 channel_num=None,
                 reduce_type=None,
                 dtype='float32',
                 name=None):
        super(WeightLSQplusQuanter, self).__init__(
            quant_bits=quant_bits,
            sign=sign,
            symmetric=symmetric,
            per_channel=per_channel,
            batch_init=batch_init,
            quant_linear=quant_linear,
            channel_num=channel_num,
            reduce_type=reduce_type,
            dtype=dtype,
            name=name)

    def _get_class(self):
        return WeightLSQplusQuanterLayer


class WeightLSQplusQuanterLayer(BaseFakeQuanterLayer):
    def __init__(self,
                 layer,
                 quant_bits=8,
                 sign=True,
                 symmetric=True,
                 per_channel=False,
                 all_postive=False,
                 batch_init=20,
                 quant_linear=False,
                 channel_num=None,
                 reduce_type=None,
                 dtype='float32',
                 name=None):
        super(WeightLSQplusQuanterLayer, self).__init__()

        self._per_channel = per_channel
        self._quant_linear = quant_linear
        self._batch_init = batch_init
        self._name = name
        self._quant_axis = 1 if quant_linear else 0
        self._collect_axis = 0 if quant_linear else 1
        self._reduce_type = reduce_type
        self.div = 2**self._quant_bits - 1
        self.qmin, self.qmax = self.qmin_qmax

        self._current_batch_id = 0
        self._init_state = 0
        scale_prefix = ("{}.scale".format(name)
                        if name else 'quant_dequant.scale')
        self._scale_name = unique_name.generate(scale_prefix)
        s_attr = ParamAttr(
            name=self._scale_name, initializer=Constant(1.0), trainable=True)

        channel_num = layer.weight.shape[
            self._quant_axis] if self._per_channel else 1

        self._scale = self.create_parameter(
            shape=[channel_num], attr=s_attr, dtype=dtype)
        self._scale.stop_gradient = False

    def init_params(self, weight):
        self.g = paddle.to_tensor(1.0 / math.sqrt(weight.numel() * self.qmax))
        if self._per_channel:
            weight_tmp = weight.detach().reshape((weight.shape[0], -1))
            mean = paddle.mean(weight_tmp, axis=self._collect_axis)
            std = paddle.std(weight_tmp, axis=self._collect_axis)
            s = paddle.max(
                paddle.stack(
                    [paddle.abs(mean - 3 * std),
                     paddle.abs(mean + 3 * std)]),
                axis=0, )
            self._scale.set_value(s / self.div)
        else:
            mean = paddle.mean(weight.detach())
            std = paddle.std(weight.detach())
            self._scale.set_value(
                max([paddle.abs(mean - 3 * std),
                     paddle.abs(mean + 3 * std)]) / self.div)
        self._init_state += 1

    def collect_gaussian(self, weight):
        if self._per_channel:
            weight_tmp = weight.detach().reshape((weight.shape[0], -1))
            mean = paddle.mean(weight_tmp, axis=self._collect_axis)
            std = paddle.std(weight_tmp, axis=self._collect_axis)
            s = paddle.max(
                paddle.stack(
                    [paddle.abs(mean - 3 * std),
                     paddle.abs(mean + 3 * std)]),
                axis=0, )
            self._scale.set_value(s * 0.9 + 0.1 * s / self.div)
        else:
            mean = paddle.mean(weight.detach())
            std = paddle.std(weight.detach())
            self._scale.set_value(self._scale * 0.9 + 0.1 * max(
                [paddle.abs(mean - 3 * std),
                 paddle.abs(mean + 3 * std)]) / self.div)
        self._init_state += 1

    def forward(self, weight):
        if self._reduce_type == "max":
            paddle.distributed.all_reduce(
                self._scale, op=paddle.distributed.ReduceOp.MAX)

        if self._init_state == 0:
            self.init_params(weight)
        elif self._init_state < self._batch_init:
            self.collect_gaussian(weight)

        weight.stop_gradient = False
        w_q = LsqFunc.apply(
            weight,
            self._scale,
            self.g,
            self.qmin,
            self.qmax,
            self._per_channel,
            self._quant_axis, )
        return w_q

    def bit_length(self):
        """ Return the bit length of quantized data.
        """
        return self._quant_bits

    def quant_axis(self):
        """ Return quantization axis.
        """
        return self._quant_axis

    def scales(self):
        """ Return output scales.
        """
        return self._scale

    def zero_points(self):
        """ Return output zero points.
        """
        if self._zero_point is None:
            if self._symmetric:
                if self._sign:
                    self._zero_point = 0
                else:
                    self._zero_point = (self.qmax + self.qmin) / 2
            else:
                self._zero_point = self.qmin - round(self.qmin / self._scale)
                self._zero_point = paddle.clip(self._zero_point, self.qmin,
                                               self.qmax)
        return self._zero_point
