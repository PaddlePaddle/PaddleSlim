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
from paddle.nn import Layer
from paddle.nn import functional as F

from paddle.nn.quant.format import ConvertibleQuantedLayer


class QuantizedRowParallelLinear(ConvertibleQuantedLayer):
    """
    The computational logic of QuantizedRowParallelLinear is the same as RowParallelLinear.
    The only difference is that its inputs are all fake quantized.
    """

    def __init__(self, layer: Layer, q_config):
        super().__init__()
        # For Linear
        self.weight = layer.weight
        self.bias = layer.bias
        self._name = layer._name
        self.input_is_parallel = layer.input_is_parallel
        self.is_mp = layer.is_mp
        self.model_parallel_group = layer.model_parallel_group
        self.linear = layer.linear

        # For FakeQuant
        self.weight_quanter = None
        self.activation_quanter = None
        if q_config.weight is not None:
            self.weight_quanter = q_config.weight._instance(layer)
        if q_config.activation is not None:
            self.activation_quanter = q_config.activation._instance(layer)

    def forward(self, input):
        quant_input = input
        quant_weight = self.weight
        if self.activation_quanter is not None:
            quant_input = self.activation_quanter(input)
        if self.weight_quanter is not None:
            quant_weight = self.weight_quanter(self.weight)
        return self._linear_forward(quant_input, quant_weight)

    def _linear_forward(self, input, weight):
        if self.input_is_parallel or (not self.is_mp):
            input_parallel = input
        else:
            # split last dim
            input_parallel = paddle.distributed.collective._c_split(
                input, group=self.model_parallel_group)

        if self.is_mp:
            output_parallel = self.linear(
                input_parallel, weight, name=self._name)
            output_ = paddle.distributed.collective._mp_allreduce(
                output_parallel,
                group=self.model_parallel_group,
                use_calc_stream=True,
                use_model_parallel=True)
            output = output_ + self.bias if self.bias is not None else output_
        else:
            output = self.linear(
                input_parallel, weight, self.bias, name=self._name)
        return output

    def weights_to_quanters(self):
        return [('weight', 'weight_quanter')]

    def activation_quanters(self):
        return ['activation_quanter']


class QuantizedColumnParallelLinear(ConvertibleQuantedLayer):
    """
    The computational logic of QuantizedColumnParallelLinear is the same as ColumnParallelLinear.
    The only difference is that its inputs are all fake quantized.
    """

    def __init__(self, layer: Layer, q_config):
        super().__init__()
        # For Linear
        self.weight = layer.weight
        self.bias = layer.bias
        self._name = layer._name
        self.is_mp = layer.is_mp
        self.model_parallel_group = layer.model_parallel_group
        self.gather_output = layer.gather_output
        self.linear = layer.linear

        # For FakeQuant
        self.weight_quanter = None
        self.activation_quanter = None
        if q_config.weight is not None:
            self.weight_quanter = q_config.weight._instance(layer)
        if q_config.activation is not None:
            self.activation_quanter = q_config.activation._instance(layer)

    def forward(self, input):
        quant_input = input
        quant_weight = self.weight
        if self.activation_quanter is not None:
            quant_input = self.activation_quanter(input)
        if self.weight_quanter is not None:
            quant_weight = self.weight_quanter(self.weight)
        return self._linear_forward(quant_input, quant_weight)

    def _linear_forward(self, input, weight):
        if self.is_mp:
            input_parallel = paddle.distributed.collective._c_identity(
                input, group=self.model_parallel_group)
        else:
            input_parallel = input

        output_parallel = self.linear(
            input_parallel, weight, self.bias, name=self._name)

        if self.gather_output and self.is_mp:
            output = paddle.distributed.collective._c_concat(
                output_parallel, group=self.model_parallel_group)
        else:
            output = output_parallel
        return output

    def weights_to_quanters(self):
        return [('weight', 'weight_quanter')]

    def activation_quanters(self):
        return ['activation_quanter']
