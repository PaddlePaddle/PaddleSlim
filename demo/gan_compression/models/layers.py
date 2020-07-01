# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, Conv2DTranspose, BatchNorm, InstanceNorm
from paddle.nn.layer import Leaky_ReLU, ReLU, Pad2D
import os
import functools

# cudnn is not better when batch size is 1.
use_cudnn = False


class conv2d(fluid.dygraph.Layer):
    """docstring for Conv2D"""

    def __init__(self,
                 num_channels,
                 num_filters=64,
                 filter_size=7,
                 stride=1,
                 stddev=0.02,
                 padding=0,
                 norm=True,
                 norm_layer=InstanceNorm,
                 relu=True,
                 relufactor=0.0,
                 use_bias=False):
        super(conv2d, self).__init__()

        if use_bias == False:
            con_bias_attr = False
        else:
            con_bias_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.0))

        self.conv = Conv2D(
            num_channels=num_channels,
            num_filters=int(num_filters),
            filter_size=int(filter_size),
            stride=stride,
            padding=padding,
            use_cudnn=use_cudnn,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=stddev)),
            bias_attr=con_bias_attr)
        if norm_layer == InstanceNorm:
            self.bn = InstanceNorm(
                num_channels=num_filters,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(1.0),
                    trainable=False),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(0.0),
                    trainable=False), )
        elif norm_layer == BatchNorm:
            self.bn = BatchNorm(
                num_channels=num_filters,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.NormalInitializer(1.0,
                                                                    0.02)),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(0.0)), )
        else:
            raise NotImplementedError

        self.relufactor = relufactor
        self.use_bias = use_bias
        self.norm = norm
        if relu:
            if relufactor == 0.0:
                self.lrelu = ReLU()
            else:
                self.lrelu = Leaky_ReLU(self.relufactor)
        self.relu = relu

    def forward(self, inputs):
        conv = self.conv(inputs)
        if self.norm:
            conv = self.bn(conv)
        if self.relu:
            conv = self.lrelu(conv)
            #conv = fluid.layers.leaky_relu(conv,alpha=self.relufactor)
        return conv


class SeparableConv2D(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 norm_layer='instance',
                 use_bias=True,
                 scale_factor=1,
                 stddev=0.02):
        super(SeparableConv2D, self).__init__()

        if use_bias == False:
            con_bias_attr = False
        else:
            con_bias_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.0))

        self.conv_sep = Conv2D(
            num_channels=num_channels,
            num_filters=num_channels * scale_factor,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            use_cudnn=use_cudnn,
            groups=num_channels,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=stddev)),
            bias_attr=con_bias_attr)

        self.norm = InstanceNorm(
            num_channels=num_filters,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(1.0)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.0)), )

        self.conv_out = Conv2D(
            num_channels=num_channels * scale_factor,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            use_cudnn=use_cudnn,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=stddev)),
            bias_attr=con_bias_attr)

    def forward(self, inputs):
        conv = self.conv_sep(inputs)
        conv = self.norm(conv)
        conv = self.conv_out(conv)
        return conv


class DeConv2D(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters=64,
                 filter_size=7,
                 stride=1,
                 stddev=0.02,
                 padding=[0, 0],
                 outpadding=[0, 0, 0, 0],
                 relu=True,
                 norm=True,
                 norm_layer=InstanceNorm,
                 relufactor=0.0,
                 use_bias=False):
        super(DeConv2D, self).__init__()

        if use_bias == False:
            de_bias_attr = False
        else:
            de_bias_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.0))

        self._deconv = Conv2DTranspose(
            num_channels,
            num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=stddev)),
            bias_attr=de_bias_attr)

        self.pad = Pad2D(paddings=outpadding, mode='constant', pad_value=0.0)
        if norm_layer == InstanceNorm:
            self.bn = InstanceNorm(
                num_channels=num_filters,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(1.0),
                    trainable=False),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(0.0),
                    trainable=False), )
        elif norm_layer == BatchNorm:
            self.bn = BatchNorm(
                num_channels=num_filters,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.NormalInitializer(1.0,
                                                                    0.02)),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(0.0)), )
        else:
            raise NotImplementedError

        self.outpadding = outpadding
        self.relufactor = relufactor
        self.use_bias = use_bias
        self.norm = norm
        self.relu = relu
        if relu:
            if relufactor == 0.0:
                self.lrelu = ReLU()
            else:
                self.lrelu = Leaky_ReLU(self.relufactor)

    def forward(self, inputs):
        #todo: add use_bias
        #if self.use_bias==False:
        conv = self._deconv(inputs)
        #else:
        #    conv = self._deconv(inputs)
        #conv = fluid.layers.pad2d(conv, paddings=self.outpadding, mode='constant', pad_value=0.0)
        conv = self.pad(conv)

        if self.norm:
            conv = self.bn(conv)
        if self.relu:
            #conv = fluid.layers.leaky_relu(conv,alpha=self.relufactor)
            conv = self.lrelu(conv)
        return conv
