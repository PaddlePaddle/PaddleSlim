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
from paddle.quantization.base_quanter import BaseQuanter


class PACTQuanter(QuanterFactory):
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

    def __init__(self,
                 quanter,
                 init_value=100.,
                 learning_rate=1000.,
                 dtype='float32',
                 name=None):
        super(PACTQuanter, self).__init__(
            quanter=quanter, init_value=init_value, dtype=dtype, name=name)

    def _get_class(self):
        return PACTQuanterLayer


class PACTQuanterLayer(BaseQuanter):
    def __init__(self,
                 layer,
                 quanter,
                 init_value=1000,
                 learning_rate=1000.,
                 dtype='float32',
                 name=None):
        super(PACTQuanterLayer, self).__init__()

        self.quanter = quanter(layer)
        alpha_prefix = ("{}.pact".format(name)
                        if name else 'quant_dequant.pact')
        name = unique_name.generate(alpha_prefix)

        alpha_attr = paddle.ParamAttr(
            name=name,
            initializer=paddle.nn.initializer.Constant(value=init_value),
            learning_rate=learning_rate)

        self.alpha = self.create_parameter(
            shape=[1], attr=alpha_attr, dtype=dtype)

    def forward(self, activation):
        out_left = paddle.nn.functional.relu(activation - self.alpha)
        out_right = paddle.nn.functional.relu(-self.alpha - activation)
        activation = activation - out_left + out_right
        return self.quanter(activation)

    def bit_length(self):
        """ Return the bit length of quantized data.
        """
        return self.quanter.bit_length()

    def quant_axis(self):
        """ Return quantization axis.
        """
        return self.quanter.quant_axis()

    def scales(self):
        """ Return output scales.
        """
        return self.quanter.scales()

    def zero_points(self):
        """ Return output zero points.
        """
        return self.quanter.zero_points()
