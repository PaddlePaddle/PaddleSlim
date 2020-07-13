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

import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Conv2DTranspose, BatchNorm, InstanceNorm, Dropout
from paddle.nn.layer import ReLU, Pad2D

__all__ = ['SeparableConv2D', 'MobileResnetBlock', 'ResnetBlock']


class SeparableConv2D(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 norm_layer=InstanceNorm,
                 use_bias=True,
                 scale_factor=1,
                 stddev=0.02,
                 use_cudnn=False):
        super(SeparableConv2D, self).__init__()

        self.conv = fluid.dygraph.LayerList([
            Conv2D(
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
                bias_attr=use_bias)
        ])

        self.conv.extend([norm_layer(num_channels * scale_factor)])

        self.conv.extend([
            Conv2D(
                num_channels=num_channels * scale_factor,
                num_filters=num_filters,
                filter_size=1,
                stride=1,
                use_cudnn=use_cudnn,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.NormalInitializer(
                        loc=0.0, scale=stddev)),
                bias_attr=use_bias)
        ])

    def forward(self, inputs):
        for sublayer in self.conv:
            inputs = sublayer(inputs)
        return inputs


class MobileResnetBlock(fluid.dygraph.Layer):
    def __init__(self, in_c, out_c, padding_type, norm_layer, dropout_rate,
                 use_bias):
        super(MobileResnetBlock, self).__init__()
        self.padding_type = padding_type
        self.dropout_rate = dropout_rate
        self.conv_block = fluid.dygraph.LayerList([])

        p = 0
        if self.padding_type == 'reflect':
            self.conv_block.extend(
                [Pad2D(
                    paddings=[1, 1, 1, 1], mode='reflect')])
        elif self.padding_type == 'replicate':
            self.conv_block.extend(
                [Pad2D(
                    inputs, paddings=[1, 1, 1, 1], mode='edge')])
        elif self.padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      self.padding_type)

        self.conv_block.extend([
            SeparableConv2D(
                num_channels=in_c,
                num_filters=out_c,
                filter_size=3,
                padding=p,
                stride=1), norm_layer(out_c), ReLU()
        ])

        self.conv_block.extend([Dropout(p=self.dropout_rate)])

        if self.padding_type == 'reflect':
            self.conv_block.extend(
                [Pad2D(
                    paddings=[1, 1, 1, 1], mode='reflect')])
        elif self.padding_type == 'replicate':
            self.conv_block.extend(
                [Pad2D(
                    inputs, paddings=[1, 1, 1, 1], mode='edge')])
        elif self.padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      self.padding_type)

        self.conv_block.extend([
            SeparableConv2D(
                num_channels=out_c,
                num_filters=in_c,
                filter_size=3,
                padding=p,
                stride=1), norm_layer(in_c)
        ])

    def forward(self, inputs):
        y = inputs
        for sublayer in self.conv_block:
            y = sublayer(y)
        out = inputs + y
        return out


class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self,
                 dim,
                 padding_type,
                 norm_layer,
                 dropout_rate,
                 use_bias=False):
        super(ResnetBlock, self).__init__()

        self.conv_block = fluid.dygraph.LayerList([])
        p = 0
        if padding_type == 'reflect':
            self.conv_block.extend(
                [Pad2D(
                    paddings=[1, 1, 1, 1], mode='reflect')])
        elif padding_type == 'replicate':
            self.conv_block.extend([Pad2D(paddings=[1, 1, 1, 1], mode='edge')])
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      padding_type)

        self.conv_block.extend([
            Conv2D(
                dim, dim, filter_size=3, padding=p, bias_attr=use_bias),
            norm_layer(dim), ReLU()
        ])
        self.conv_block.extend([Dropout(dropout_rate)])

        p = 0
        if padding_type == 'reflect':
            self.conv_block.extend(
                [Pad2D(
                    paddings=[1, 1, 1, 1], mode='reflect')])
        elif padding_type == 'replicate':
            self.conv_block.extend([Pad2D(paddings=[1, 1, 1, 1], mode='edge')])
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      padding_type)

        self.conv_block.extend([
            Conv2D(
                dim, dim, filter_size=3, padding=p, bias_attr=use_bias),
            norm_layer(dim)
        ])

    def forward(self, inputs):
        y = inputs
        for sublayer in self.conv_block:
            y = sublayer(y)
        return y + inputs

