# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout, _ConvNd
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D

import math

__all__ = [
    "MobileNetV2_x0_25", "MobileNetV2_x0_5", "MobileNetV2_x0_75", "MobileNetV2",
    "MobileNetV2_x1_5", "MobileNetV2_x2_0"
]


class Sparse_Conv2D(_ConvNd):
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
                 data_format="NCHW",
                 group_size=16,
                 target_rate=0,
                 drop_rate=0):
        super(Sparse_Conv2D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            False,
            2,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)
        self.group_size = group_size
        self.target_rate = target_rate
        self.drop_rate = drop_rate

    def forward(self, x):
        if self._padding_mode != 'zeros':
            x = F.pad(x,
                      self._reversed_padding_repeated_twice,
                      mode=self._padding_mode,
                      data_format=self._data_format)
        with paddle.no_grad():
            weight = self.weight.clone()
            weight = weight.abs()
            weight = weight.reshape([-1, self.group_size])
            threshold = paddle.sort(
                weight, axis=1)[:, self.target_rate].expand(
                    [weight.shape[1], weight.shape[0]]).t()
            drop_mask = (weight >= threshold)
            drop_mask = drop_mask.reshape(self.weight.shape)

        if self.training:
            with paddle.no_grad():
                bernoulli_mask = paddle.full(drop_mask.shape, self.drop_rate)
                bernoulli_mask = paddle.bernoulli(bernoulli_mask)
                bernoulli_mask = paddle.less_equal(
                    bernoulli_mask.astype(int),
                    (bernoulli_mask * drop_mask).astype(int)).astype(bool)
            weight = self.weight * bernoulli_mask
        else:
            weight = self.weight * drop_mask
        out = F.conv._conv_nd(
            x,
            weight,
            bias=self.bias,
            stride=self._stride,
            padding=self._updated_padding,
            padding_algorithm=self._padding_algorithm,
            dilation=self._dilation,
            groups=self._groups,
            data_format=self._data_format,
            channel_dim=self._channel_dim,
            op_type=self._op_type,
            use_cudnn=self._use_cudnn)
        return out

    def set_target_rate(self, target_rate):
        self.target_rate = target_rate

    def set_drop_rate(self, drop_rate):
        self.drop_rate = drop_rate

    def get_infer_sparsity(self):
        with paddle.no_grad():
            weight = self.weight.clone()
            weight = weight.abs()
            weight = weight.reshape([-1, self.group_size])
            threshold = paddle.sort(
                weight, axis=1)[:, self.target_rate].expand(
                    [weight.shape[1], weight.shape[0]]).t()
            drop_mask = (weight >= threshold)
            drop_mask = drop_mask.reshape(self.weight.shape)
            weight = self.weight * drop_mask
            sparsity = float(paddle.sum((weight == 0).astype(int))) / float(
                weight.numel())
        return sparsity

    def get_real_sparsity(self, threashold=1e-6):
        with paddle.no_grad():
            weight = self.weight.clone()
            weight = float(
                paddle.sum(((weight - 0) < threashold).astype(int))) / float(
                    weight.numel())
        return weight

    def print_grad(self):
        print(self.weight.gradient)


class Sparse_ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 channels=None,
                 num_groups=1,
                 name=None,
                 use_cudnn=True):
        super(Sparse_ConvBNLayer, self).__init__()

        self._conv = Sparse_Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)

        self._batch_norm = BatchNorm(
            num_filters,
            param_attr=ParamAttr(name=name + "_bn_scale"),
            bias_attr=ParamAttr(name=name + "_bn_offset"),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")

    def forward(self, inputs, if_act=True):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if if_act:
            y = F.relu6(y)
        return y


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 channels=None,
                 num_groups=1,
                 name=None,
                 use_cudnn=True):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)

        self._batch_norm = BatchNorm(
            num_filters,
            param_attr=ParamAttr(name=name + "_bn_scale"),
            bias_attr=ParamAttr(name=name + "_bn_offset"),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")

    def forward(self, inputs, if_act=True):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if if_act:
            y = F.relu6(y)
        return y


class InvertedResidualUnit(nn.Layer):
    def __init__(self, num_channels, num_in_filter, num_filters, stride,
                 filter_size, padding, expansion_factor, name):
        super(InvertedResidualUnit, self).__init__()
        num_expfilter = int(round(num_in_filter * expansion_factor))
        self._expand_conv = Sparse_ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            name=name + "GroupSparse_expand")

        self._bottleneck_conv = ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            num_groups=num_expfilter,
            use_cudnn=False,
            name=name + "_dwise")

        self._linear_conv = Sparse_ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            name=name + "GroupSparse_linear")

    def forward(self, inputs, ifshortcut):
        y = self._expand_conv(inputs, if_act=True)
        y = self._bottleneck_conv(y, if_act=True)
        y = self._linear_conv(y, if_act=False)
        if ifshortcut:
            y = paddle.add(inputs, y)
        return y


class InvresiBlocks(nn.Layer):
    def __init__(self, in_c, t, c, n, s, name):
        super(InvresiBlocks, self).__init__()

        self._first_block = InvertedResidualUnit(
            num_channels=in_c,
            num_in_filter=in_c,
            num_filters=c,
            stride=s,
            filter_size=3,
            padding=1,
            expansion_factor=t,
            name=name + "_1")

        self._block_list = []
        for i in range(1, n):
            block = self.add_sublayer(
                name + "_" + str(i + 1),
                sublayer=InvertedResidualUnit(
                    num_channels=c,
                    num_in_filter=c,
                    num_filters=c,
                    stride=1,
                    filter_size=3,
                    padding=1,
                    expansion_factor=t,
                    name=name + "_" + str(i + 1)))
            self._block_list.append(block)

    def forward(self, inputs):
        y = self._first_block(inputs, ifshortcut=False)
        for block in self._block_list:
            y = block(y, ifshortcut=True)
        return y


class MobileNet(nn.Layer):
    def __init__(self, class_dim=1000, scale=1.0):
        super(MobileNet, self).__init__()
        self.scale = scale
        self.class_dim = class_dim

        bottleneck_params_list = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        self.conv1 = ConvBNLayer(
            num_channels=3,
            num_filters=int(32 * scale),
            filter_size=3,
            stride=2,
            padding=1,
            name="conv1_1")

        self.block_list = []
        i = 1
        in_c = int(32 * scale)
        for layer_setting in bottleneck_params_list:
            t, c, n, s = layer_setting
            i += 1
            block = self.add_sublayer(
                "conv" + str(i),
                sublayer=InvresiBlocks(
                    in_c=in_c,
                    t=t,
                    c=int(c * scale),
                    n=n,
                    s=s,
                    name="conv" + str(i)))
            self.block_list.append(block)
            in_c = int(c * scale)

        self.out_c = int(1280 * scale) if scale > 1.0 else 1280
        self.conv9 = ConvBNLayer(
            num_channels=in_c,
            num_filters=self.out_c,
            filter_size=1,
            stride=1,
            padding=0,
            name="conv9")

        self.pool2d_avg = AdaptiveAvgPool2D(1)

        self.out = Linear(
            self.out_c,
            class_dim,
            weight_attr=ParamAttr(name="fc10_weights"),
            bias_attr=ParamAttr(name="fc10_offset"))

    def forward(self, inputs):
        y = self.conv1(inputs, if_act=True)
        for block in self.block_list:
            y = block(y)
        y = self.conv9(y, if_act=True)
        y = self.pool2d_avg(y)
        y = paddle.flatten(y, start_axis=1, stop_axis=-1)
        y = self.out(y)
        return y


def MobileNetV2_x0_25(**args):
    model = MobileNet(scale=0.25, **args)
    return model


def MobileNetV2_x0_5(**args):
    model = MobileNet(scale=0.5, **args)
    return model


def MobileNetV2_x0_75(**args):
    model = MobileNet(scale=0.75, **args)
    return model


def MobileNetV2(**args):
    model = MobileNet(scale=1.0, **args)
    return model


def MobileNetV2_x1_5(**args):
    model = MobileNet(scale=1.5, **args)
    return model


def MobileNetV2_x2_0(**args):
    model = MobileNet(scale=2.0, **args)
    return model
