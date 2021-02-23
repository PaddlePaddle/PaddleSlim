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
from paddle.nn.initializer import KaimingNormal
import math

__all__ = [
    "MobileNetV1_x0_25", "MobileNetV1_x0_5", "MobileNetV1_x0_75", "MobileNetV1"
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
                 act='relu',
                 name=None):
        super(Sparse_ConvBNLayer, self).__init__()

        self._conv = Sparse_Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(
                initializer=KaimingNormal(), name=name + "_weights"),
            bias_attr=False)

        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name + "_bn_scale"),
            bias_attr=ParamAttr(name + "_bn_offset"),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
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
                 act='relu',
                 name=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(
                initializer=KaimingNormal(), name=name + "_weights"),
            bias_attr=False)

        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name + "_bn_scale"),
            bias_attr=ParamAttr(name + "_bn_offset"),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class DepthwiseSeparable(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters1,
                 num_filters2,
                 num_groups,
                 stride,
                 scale,
                 name=None):
        super(DepthwiseSeparable, self).__init__()

        self._depthwise_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=int(num_filters1 * scale),
            filter_size=3,
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            name=name + "_dw")

        self._pointwise_conv = Sparse_ConvBNLayer(
            num_channels=int(num_filters1 * scale),
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0,
            name=name + "_sep")

    def forward(self, inputs):
        y = self._depthwise_conv(inputs)
        y = self._pointwise_conv(y)
        return y


class MobileNet(nn.Layer):
    def __init__(self, scale=1.0, class_dim=1000):
        super(MobileNet, self).__init__()
        self.scale = scale
        self.block_list = []

        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            channels=3,
            num_filters=int(32 * scale),
            stride=2,
            padding=1,
            name="conv1")

        conv2_1 = self.add_sublayer(
            "conv2_1",
            sublayer=DepthwiseSeparable(
                num_channels=int(32 * scale),
                num_filters1=32,
                num_filters2=64,
                num_groups=32,
                stride=1,
                scale=scale,
                name="conv2_1"))
        self.block_list.append(conv2_1)

        conv2_2 = self.add_sublayer(
            "conv2_2",
            sublayer=DepthwiseSeparable(
                num_channels=int(64 * scale),
                num_filters1=64,
                num_filters2=128,
                num_groups=64,
                stride=2,
                scale=scale,
                name="conv2_2"))
        self.block_list.append(conv2_2)

        conv3_1 = self.add_sublayer(
            "conv3_1",
            sublayer=DepthwiseSeparable(
                num_channels=int(128 * scale),
                num_filters1=128,
                num_filters2=128,
                num_groups=128,
                stride=1,
                scale=scale,
                name="conv3_1"))
        self.block_list.append(conv3_1)

        conv3_2 = self.add_sublayer(
            "conv3_2",
            sublayer=DepthwiseSeparable(
                num_channels=int(128 * scale),
                num_filters1=128,
                num_filters2=256,
                num_groups=128,
                stride=2,
                scale=scale,
                name="conv3_2"))
        self.block_list.append(conv3_2)

        conv4_1 = self.add_sublayer(
            "conv4_1",
            sublayer=DepthwiseSeparable(
                num_channels=int(256 * scale),
                num_filters1=256,
                num_filters2=256,
                num_groups=256,
                stride=1,
                scale=scale,
                name="conv4_1"))
        self.block_list.append(conv4_1)

        conv4_2 = self.add_sublayer(
            "conv4_2",
            sublayer=DepthwiseSeparable(
                num_channels=int(256 * scale),
                num_filters1=256,
                num_filters2=512,
                num_groups=256,
                stride=2,
                scale=scale,
                name="conv4_2"))
        self.block_list.append(conv4_2)

        for i in range(5):
            conv5 = self.add_sublayer(
                "conv5_" + str(i + 1),
                sublayer=DepthwiseSeparable(
                    num_channels=int(512 * scale),
                    num_filters1=512,
                    num_filters2=512,
                    num_groups=512,
                    stride=1,
                    scale=scale,
                    name="conv5_" + str(i + 1)))
            self.block_list.append(conv5)

        conv5_6 = self.add_sublayer(
            "conv5_6",
            sublayer=DepthwiseSeparable(
                num_channels=int(512 * scale),
                num_filters1=512,
                num_filters2=1024,
                num_groups=512,
                stride=2,
                scale=scale,
                name="conv5_6"))
        self.block_list.append(conv5_6)

        conv6 = self.add_sublayer(
            "conv6",
            sublayer=DepthwiseSeparable(
                num_channels=int(1024 * scale),
                num_filters1=1024,
                num_filters2=1024,
                num_groups=1024,
                stride=1,
                scale=scale,
                name="conv6"))
        self.block_list.append(conv6)

        self.pool2d_avg = AdaptiveAvgPool2D(1)

        self.out = Linear(
            int(1024 * scale),
            class_dim,
            weight_attr=ParamAttr(
                initializer=KaimingNormal(), name="fc7_weights"),
            bias_attr=ParamAttr(name="fc7_offset"))

    def forward(self, inputs):
        y = self.conv1(inputs)
        for block in self.block_list:
            y = block(y)
        y = self.pool2d_avg(y)
        y = paddle.flatten(y, start_axis=1, stop_axis=-1)
        y = self.out(y)
        return y


def MobileNetV1_x0_25(**args):
    model = MobileNet(scale=0.25, **args)
    return model


def MobileNetV1_x0_5(**args):
    model = MobileNet(scale=0.5, **args)
    return model


def MobileNetV1_x0_75(**args):
    model = MobileNet(scale=0.75, **args)
    return model


def MobileNetV1(**args):
    model = MobileNet(scale=1.0, **args)
    return model
