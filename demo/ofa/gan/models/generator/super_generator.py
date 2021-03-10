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
import functools
import paddle
import paddle.nn as nn
from paddle.nn import BatchNorm2D, InstanceNorm2D, Dropout, ReLU, Pad2D
from paddleslim.nas.ofa.layers import SuperConv2D, SuperConv2DTranspose, SuperSeparableConv2D


class SuperMobileResnetBlock(nn.Layer):
    def __init__(self, dim, padding_type, norm_layer, dropout_rate, use_bias):
        super(SuperMobileResnetBlock, self).__init__()
        self.conv_block = nn.LayerList([])
        p = 0
        if padding_type == 'reflect':
            self.conv_block.extend(
                [Pad2D(
                    padding=[1, 1, 1, 1], mode="reflect")])
        elif padding_type == 'replicate':
            self.conv_block.extend(
                [Pad2D(
                    padding=[1, 1, 1, 1], mode="replicate")])
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      self.padding_type)

        self.conv_block.extend([
            SuperSeparableConv2D(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                stride=1,
                padding=p,
                norm_layer=norm_layer), norm_layer(dim), ReLU()
        ])
        self.conv_block.extend([Dropout(dropout_rate)])

        p = 0
        if padding_type == 'reflect':
            self.conv_block.extend(
                [Pad2D(
                    padding=[1, 1, 1, 1], mode="reflect")])
        elif padding_type == 'replicate':
            self.conv_block.extend(
                [Pad2D(
                    padding=[1, 1, 1, 1], mode="replicate")])
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      self.padding_type)

        self.conv_block.extend([
            SuperSeparableConv2D(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                stride=1,
                padding=p,
                norm_layer=norm_layer), norm_layer(dim)
        ])

    def forward(self, input, channel):
        x = input
        cnt = 0
        for sublayer in self.conv_block:
            if isinstance(sublayer, SuperSeparableConv2D):
                if cnt == 1:
                    channel = input.shape[1]
                x = sublayer(x, channel=channel)
                cnt += 1
            else:
                x = sublayer(x)
        out = input + x
        return out


class SuperMobileResnetGenerator(nn.Layer):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf,
                 norm_layer=InstanceNorm2D,
                 dropout_rate=0,
                 n_blocks=6,
                 padding_type='reflect'):
        assert n_blocks >= 0
        super(SuperMobileResnetGenerator, self).__init__()
        use_bias = norm_layer == InstanceNorm2D

        self.model = nn.LayerList([])
        self.model.extend([
            Pad2D(
                padding=[3, 3, 3, 3], mode="reflect"), SuperConv2D(
                    input_nc, ngf, kernel_size=7, padding=0,
                    bias_attr=use_bias), norm_layer(ngf), ReLU()
        ])

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            self.model.extend([
                SuperConv2D(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias_attr=use_bias), norm_layer(int(ngf * mult * 2)), ReLU()
            ])

        mult = 2**n_downsampling
        n_blocks1 = n_blocks // 3
        n_blocks2 = n_blocks1
        n_blocks3 = n_blocks - n_blocks1 - n_blocks2

        for i in range(n_blocks1):
            self.model.extend([
                SuperMobileResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    dropout_rate=dropout_rate,
                    use_bias=use_bias)
            ])

        for i in range(n_blocks2):
            self.model.extend([
                SuperMobileResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    dropout_rate=dropout_rate,
                    use_bias=use_bias)
            ])

        for i in range(n_blocks3):
            self.model.extend([
                SuperMobileResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    dropout_rate=dropout_rate,
                    use_bias=use_bias)
            ])

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            self.model.extend([
                SuperConv2DTranspose(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    output_padding=1,
                    stride=2,
                    padding=1,
                    bias_attr=use_bias), norm_layer(int(ngf * mult / 2)), ReLU()
            ])

        self.model.extend([Pad2D(padding=[3, 3, 3, 3], mode="reflect")])
        self.model.extend(
            [SuperConv2D(
                ngf, output_nc, kernel_size=7, padding=0)])

    def forward(self, input):
        configs = self.configs
        x = paddle.clip(input, min=-1, max=1)
        cnt = 0
        for i in range(0, 10):
            sublayer = self.model[i]
            if isinstance(sublayer, SuperConv2D):
                channel = configs['channel'][cnt] * (2**cnt)
                config = {'channel': channel}
                x = sublayer(x, **config)
                cnt += 1
            else:
                x = sublayer(x)

        for i in range(3):
            for j in range(10 + i * 3, 13 + i * 3):
                if len(configs['channel']) == 6:
                    channel = configs['channel'][3] * 4
                else:
                    channel = configs['channel'][i + 3] * 4
                config = {'channel': channel}
                sublayer = self.model[j]
                x = sublayer(x, **config)

        cnt = 2
        for i in range(19, 27):
            sublayer = self.model[i]
            if isinstance(sublayer, SuperConv2DTranspose):
                cnt -= 1
                if len(configs['channel']) == 6:
                    channel = configs['channel'][5 - cnt] * (2**cnt)
                else:
                    channel = configs['channel'][7 - cnt] * (2**cnt)
                config = {'channel': channel}
                x = sublayer(x, **config)
            elif isinstance(sublayer, SuperConv2D):
                config = {'channel': sublayer._out_channels}
                x = sublayer(x, **config)
            else:
                x = sublayer(x)
        x = paddle.tanh(x)
        return x
