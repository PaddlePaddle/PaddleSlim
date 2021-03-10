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
from paddle.nn import InstanceNorm2D, Conv2D, Conv2DTranspose, ReLU, Pad2D
from .modules import SeparableConv2D, MobileResnetBlock


class SubMobileResnetGenerator(paddle.nn.Layer):
    def __init__(self,
                 input_nc,
                 output_nc,
                 config,
                 norm_layer=InstanceNorm2D,
                 dropout_rate=0,
                 n_blocks=9,
                 padding_type='reflect'):
        super(SubMobileResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == InstanceNorm2D
        else:
            use_bias = norm_layer == InstanceNorm2D

        self.model = nn.LayerList([
            Pad2D(
                padding=[3, 3, 3, 3], mode="reflect"), Conv2D(
                    input_nc,
                    config['channels'][0],
                    kernel_size=7,
                    padding=0,
                    bias_attr=use_bias), norm_layer(config['channels'][0]),
            ReLU()
        ])

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            in_c = config['channels'][i]
            out_c = config['channels'][i + 1]
            self.model.extend([
                Conv2D(
                    in_c * mult,
                    out_c * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias_attr=use_bias), norm_layer(out_c * mult * 2), ReLU()
            ])

        mult = 2**n_downsampling

        in_c = config['channels'][2]
        for i in range(n_blocks):
            if len(config['channels']) == 6:
                offset = 0
            else:
                offset = i // 3
            out_c = config['channels'][offset + 3]
            self.model.extend([
                MobileResnetBlock(
                    in_c * mult,
                    out_c * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    dropout_rate=dropout_rate,
                    use_bias=use_bias)
            ])

        if len(config['channels']) == 6:
            offset = 4
        else:
            offset = 6
        for i in range(n_downsampling):
            out_c = config['channels'][offset + i]
            mult = 2**(n_downsampling - i)
            output_size = (i + 1) * (self.cfgs.crop_size / 2)
            self.model.extend([
                Conv2DTranspose(
                    in_c * mult,
                    int(out_c * mult / 2),
                    kernel_size=3,
                    output_size=output_size,
                    stride=2,
                    padding=1,
                    bias_attr=use_bias), norm_layer(int(out_c * mult / 2)),
                ReLU()
            ])
            in_c = out_c

        self.model.extend([Pad2D(padding=[3, 3, 3, 3], mode="reflect")])
        self.model.extend([Conv2D(in_c, output_nc, kernel_size=7, padding=0)])

    def forward(self, inputs):
        y = nn.clip(input, min=-1, max=1)
        for sublayer in self.model:
            y = sublayer(y)
        y = paddle.tanh(y)
        return y
