#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import numpy as np

import paddle
import paddle.nn as nn
from paddle.nn import Conv2D, Conv2DTranspose, BatchNorm2D, InstanceNorm2D, Dropout
from paddle.nn.layer import LeakyReLU, ReLU, Pad2D

__all__ = ['SeparableConv2D', 'MobileResnetBlock', 'ResnetBlock']


class SeparableConv2D(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 norm_layer=InstanceNorm2D,
                 use_bias=True,
                 scale_factor=1,
                 stddev=0.02):
        super(SeparableConv2D, self).__init__()

        self.conv = nn.LayerList([
            Conv2D(
                in_channels=in_channels,
                out_channels=in_channels * scale_factor,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(
                    mean=0.0, std=stddev)),
                bias_attr=use_bias)
        ])

        self.conv.extend([norm_layer(in_channels * scale_factor)])

        self.conv.extend([
            Conv2D(
                in_channels=in_channels * scale_factor,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(
                    mean=0.0, std=stddev)),
                bias_attr=use_bias)
        ])

    def forward(self, inputs):
        for sublayer in self.conv:
            inputs = sublayer(inputs)
        return inputs


class MobileResnetBlock(nn.Layer):
    def __init__(self, in_c, out_c, padding_type, norm_layer, dropout_rate,
                 use_bias):
        super(MobileResnetBlock, self).__init__()
        self.padding_type = padding_type
        self.dropout_rate = dropout_rate
        self.conv_block = nn.LayerList([])

        p = 0
        if self.padding_type == 'reflect':
            self.conv_block.extend(
                [Pad2D(
                    padding=[1, 1, 1, 1], mode='reflect')])
        elif self.padding_type == 'replicate':
            self.conv_block.extend(
                [Pad2D(
                    padding=[1, 1, 1, 1], mode='replicate')])
        elif self.padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      self.padding_type)

        self.conv_block.extend([
            SeparableConv2D(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=3,
                padding=p,
                stride=1), norm_layer(out_c), ReLU()
        ])

        self.conv_block.extend([Dropout(p=self.dropout_rate)])

        if self.padding_type == 'reflect':
            self.conv_block.extend(
                [Pad2D(
                    padding=[1, 1, 1, 1], mode='reflect')])
        elif self.padding_type == 'replicate':
            self.conv_block.extend(
                [Pad2D(
                    paddings=[1, 1, 1, 1], mode='replicate')])
        elif self.padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      self.padding_type)

        self.conv_block.extend([
            SeparableConv2D(
                in_channels=out_c,
                out_channels=in_c,
                kernel_size=3,
                padding=p,
                stride=1), norm_layer(in_c)
        ])

    def forward(self, inputs):
        y = inputs
        for sublayer in self.conv_block:
            y = sublayer(y)
        out = inputs + y
        return out


class ResnetBlock(nn.Layer):
    def __init__(self,
                 dim,
                 padding_type,
                 norm_layer,
                 dropout_rate,
                 use_bias=False):
        super(ResnetBlock, self).__init__()

        self.conv_block = nn.LayerList([])
        p = 0
        if padding_type == 'reflect':
            self.conv_block.extend(
                [Pad2D(
                    padding=[1, 1, 1, 1], mode='reflect')])
        elif padding_type == 'replicate':
            self.conv_block.extend(
                [Pad2D(
                    padding=[1, 1, 1, 1], mode='replicate')])
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      padding_type)

        self.conv_block.extend([
            Conv2D(
                dim, dim, kernel_size=3, padding=p, bias_attr=use_bias),
            norm_layer(dim), ReLU()
        ])
        self.conv_block.extend([Dropout(dropout_rate)])

        p = 0
        if padding_type == 'reflect':
            self.conv_block.extend(
                [Pad2D(
                    padding=[1, 1, 1, 1], mode='reflect')])
        elif padding_type == 'replicate':
            self.conv_block.extend(
                [Pad2D(
                    padding=[1, 1, 1, 1], mode='replicate')])
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      padding_type)

        self.conv_block.extend([
            Conv2D(
                dim, dim, kernel_size=3, padding=p, bias_attr=use_bias),
            norm_layer(dim)
        ])

    def forward(self, inputs):
        y = inputs
        for sublayer in self.conv_block:
            y = sublayer(y)
        return y + inputs
