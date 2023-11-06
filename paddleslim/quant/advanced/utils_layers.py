# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn.initializer import Constant
from paddle.distributed.fleet.meta_parallel import ColumnParallelLinear, RowParallelLinear

__all__ = ['ShiftSmoothHelpLayer', 'WOBiasHelpLayer']


class ShiftSmoothHelpLayer(nn.Layer):
    def __init__(self, layer):
        super(ShiftSmoothHelpLayer, self).__init__()
        self.weight = layer.weight
        shift_shape = self.weight.shape[0]
        if not hasattr(layer, "bias") or layer.bias is None:
            self.bias = paddle.create_parameter(
                shape=[self.weight.shape[1]],
                dtype=self.weight.dtype,
                default_initializer=paddle.nn.initializer.Constant(0.0),
                is_bias=True, )
            layer.bias = self.bias
        self.layer = layer
        self.layer_type = type(layer)
        # add
        self.shift_bias = self.create_parameter(
            shape=[shift_shape],
            attr=ParamAttr(initializer=Constant(value=0.)),
            dtype=self.weight.dtype)
        # multiply
        self.smooth_weight = self.create_parameter(
            shape=[shift_shape],
            attr=ParamAttr(initializer=Constant(value=1.)),
            dtype=self.weight.dtype)

    def forward(self, input):
        shift_input = input
        shift_input = paddle.add(shift_input, self.shift_bias)
        smooth_input = paddle.multiply(shift_input, self.smooth_weight)
        return self.layer(smooth_input)

    def convert_weight(self, shift_bias=None, smooth_weight=None):
        # shift
        if shift_bias is not None:
            shift_bias = shift_bias.cast(self.weight.dtype)
            self.shift_bias.set_value(-shift_bias)
            shift_linear_bias = paddle.matmul(shift_bias, self.weight)

            if self.layer_type == RowParallelLinear:
                parallel_shift_linear_bias = paddle.distributed.collective._mp_allreduce(
                    shift_linear_bias,
                    group=self.layer.model_parallel_group,
                    use_calc_stream=True,
                    use_model_parallel=True)
                self.bias.set_value(self.bias + parallel_shift_linear_bias)
            else:
                self.bias.set_value(self.bias + shift_linear_bias)

        # smooth
        if smooth_weight is not None:
            self.smooth_weight.set_value(
                1 / smooth_weight.squeeze().cast(self.weight.dtype))
            self.weight.set_value(
                self.weight * smooth_weight.transpose(perm=[1, 0]))


class WOBiasHelpLayer(nn.Layer):
    def __init__(self, layer):
        super(WOBiasHelpLayer, self).__init__()
        self.weight = layer.weight
        self.bias = paddle.create_parameter(
            shape=self.weight.shape,
            dtype=self.weight.dtype,
            default_initializer=paddle.nn.initializer.Constant(0.0),
            is_bias=True, )
        self.layer = layer

    def forward(self, input):
        return self.layer(input) + self.bias
