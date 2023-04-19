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
from paddle import nn
from paddle.autograd import PyLayer
from paddle.nn import functional as F
from paddle.nn.layer import Conv1D
from paddle.nn.layer.common import Linear


class BinaryQuantizer(PyLayer):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = paddle.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensor()[0]
        grad_input = grad_output
        grad_input[input >= 1] = 0
        grad_input[input <= -1] = 0
        return grad_input.clone()


class BiLinear(Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 weight_attr=None,
                 bias_attr=None,
                 name=None):
        super(BiLinear, self).__init__(
            in_features,
            out_features,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            name=name)

        self.scale_weight_init = False
        self.scale_weight = paddle.create_parameter(shape=[1], dtype='float32')

    def forward(self, input):
        ba = input

        bw = self.weight
        bw = bw - bw.mean()

        if self.scale_weight_init == False:
            scale_weight = F.linear(ba, bw).std() / F.linear(
                paddle.sign(ba), paddle.sign(bw)).std()
            if paddle.isnan(scale_weight):
                scale_weight = bw.std() / paddle.sign(bw).std()
            self.scale_weight.set_value(scale_weight)
            self.scale_weight_init = True

        ba = BinaryQuantizer.apply(ba)
        bw = BinaryQuantizer.apply(bw)
        bw = bw * self.scale_weight

        out = F.linear(x=ba, weight=bw, bias=self.bias, name=self.name)
        return out


class BiConv1D(Conv1D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCL"):
        super(BiConv1D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, padding_mode, weight_attr, bias_attr, data_format)
        self.scale_weight_init = False
        self.scale_weight = paddle.create_parameter(shape=[1], dtype='float32')

    def forward(self, input):
        ba = input

        bw = self.weight
        bw = bw - bw.mean()

        padding = 0
        if self._padding_mode != "zeros":
            ba = F.pad(
                ba,
                self._reversed_padding_repeated_twice,
                mode=self._padding_mode,
                data_format=self._data_format)
        else:
            padding = self._padding

        if self.scale_weight_init == False:
            scale_weight = F.conv1d(ba, bw, bias=self.bias, padding=padding, stride=self._stride, dilation=self._dilation, groups=self._groups, data_format=self._data_format).std() / \
                F.conv1d(paddle.sign(ba), paddle.sign(bw), bias=self.bias, padding=padding, stride=self._stride, dilation=self._dilation, groups=self._groups, data_format=self._data_format).std()
            if paddle.isnan(scale_weight):
                scale_weight = bw.std() / paddle.sign(bw).std()

            self.scale_weight.set_value(scale_weight)
            self.scale_weight_init = True

        ba = BinaryQuantizer.apply(ba)
        bw = BinaryQuantizer.apply(bw)
        bw = bw * self.scale_weight

        return F.conv1d(
            ba,
            bw,
            bias=self.bias,
            padding=padding,
            stride=self._stride,
            dilation=self._dilation,
            groups=self._groups,
            data_format=self._data_format)


def _to_bi_function(model, fp_layers=[]):
    for name, layer in model.named_children():
        if id(layer) in fp_layers:
            continue
        if isinstance(layer, Linear):
            new_layer = BiLinear(layer.weight.shape[0], layer.weight.shape[1],
                                 layer._weight_attr, layer._bias_attr,
                                 layer.name)
            new_layer.weight = layer.weight
            new_layer.bias = layer.bias
            model._sub_layers[name] = new_layer
        elif isinstance(layer, Conv1D):
            new_layer = BiConv1D(layer._in_channels, layer._out_channels,
                                 layer._kernel_size, layer._stride,
                                 layer._padding, layer._dilation, layer._groups,
                                 layer._padding_mode, layer._param_attr,
                                 layer._bias_attr, layer._data_format)
            new_layer.weight = layer.weight
            new_layer.bias = layer.bias
            model._sub_layers[name] = new_layer
        elif isinstance(layer, nn.ReLU):
            model._sub_layers[name] = nn.Hardtanh()
        else:
            model._sub_layers[name] = _to_bi_function(layer, fp_layers)
    return model
