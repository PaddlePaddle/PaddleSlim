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
from paddle.nn import InstanceNorm2D, Conv2D, Conv2DTranspose, ReLU, Pad2D
from .modules import ResnetBlock


class ResnetGenerator(nn.Layer):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf,
                 norm_layer=InstanceNorm2D,
                 dropout_rate=0,
                 n_blocks=6,
                 padding_type='reflect'):
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == InstanceNorm2D
        else:
            use_bias = norm_layer == InstanceNorm2D

        self.model = nn.LayerList([
            Pad2D(
                padding=[3, 3, 3, 3], mode="reflect"), Conv2D(
                    input_nc,
                    int(ngf),
                    kernel_size=7,
                    padding=0,
                    bias_attr=use_bias), norm_layer(ngf), ReLU()
        ])

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            self.model.extend([
                Conv2D(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias_attr=use_bias), norm_layer(ngf * mult * 2), ReLU()
            ])

        mult = 2**n_downsampling
        for i in range(n_blocks):
            self.model.extend([
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    dropout_rate=dropout_rate,
                    use_bias=use_bias)
            ])

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            output_size = (i + 1) * (self.cfgs.crop_size / 2)
            self.model.extend([
                Conv2DTranspose(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_size=output_size,
                    bias_attr=use_bias), Pad2D(
                        padding=[0, 1, 0, 1], mode='constant', pad_value=0.0),
                norm_layer(int(ngf * mult / 2)), ReLU()
            ])

        self.model.extend([Pad2D(padding=[3, 3, 3, 3], mode="reflect")])
        self.model.extend([Conv2D(ngf, output_nc, kernel_size=7, padding=0)])

    def forward(self, inputs):
        y = paddle.clip(inputs, min=-1.0, max=1.0)
        for sublayer in self.model:
            y = sublayer(y)
        y = paddle.tanh(y)
        return y
