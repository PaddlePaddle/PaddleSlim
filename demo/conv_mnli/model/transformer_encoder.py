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
"dygraph transformer layers"

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import Iterable

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Embedding, LayerNorm, Linear, Layer, Conv2D, BatchNorm, Pool2D, to_variable
from paddle.fluid.initializer import NormalInitializer


class ConvBNRelu(fluid.dygraph.Layer):
    def __init__(self,
                 in_c=768,
                 out_c=768,
                 filter_size=[3, 1],
                 dilation=1,
                 is_test=False,
                 use_cudnn=True,
                 name=None):
        super(ConvBNRelu, self).__init__()
        self._name = name
        conv_std = (2.0 /
                    (filter_size[0] * filter_size[1] * out_c * in_c))**0.5
        conv_param = fluid.ParamAttr(
            name=name if name is None else (name + "_conv.weights"),
            initializer=fluid.initializer.Normal(0.0, conv_std))

        self.conv = Conv2D(
            in_c,
            out_c,
            filter_size,
            dilation=[dilation, 1],
            padding=[(filter_size[0] - 1) * dilation // 2, 0],
            param_attr=conv_param,
            act=None,
            bias_attr=False,
            use_cudnn=use_cudnn)
        self.bn = BatchNorm(out_c, act="relu", is_test=False)

    def forward(self, inputs):
        conv = self.conv(inputs)
        bn = self.bn(conv)
        return conv


class GateConv(fluid.dygraph.Layer):
    def __init__(self,
                 in_c=768,
                 out_c=768,
                 filter_size=[3, 1],
                 dilation=1,
                 is_test=False,
                 use_cudnn=True,
                 name=None):
        super(GateConv, self).__init__()
        conv_std = (2.0 /
                    (filter_size[0] * filter_size[1] * out_c * in_c))**0.5
        conv_param = fluid.ParamAttr(
            name=name if name is None else (name + "_conv.weights"),
            initializer=fluid.initializer.Normal(0.0, conv_std))

        gate_param = fluid.ParamAttr(
            name=name if name is None else (name + "_conv_gate.weights"),
            initializer=fluid.initializer.Normal(0.0, conv_std))

        self.conv = Conv2D(
            in_c,
            out_c,
            filter_size,
            dilation=[dilation, 1],
            padding=[(filter_size[0] - 1) * dilation // 2, 0],
            param_attr=conv_param,
            act=None,
            use_cudnn=use_cudnn)

        self.gate = Conv2D(
            in_c,
            out_c,
            filter_size,
            dilation=[dilation, 1],
            padding=[(filter_size[0] - 1) * dilation // 2, 0],
            param_attr=gate_param,
            act="sigmoid",
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        conv = self.conv(inputs)
        gate = self.gate(inputs)
        return conv * gate


class ConvBNBlock(Layer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 dillation=1,
                 block_size=4,
                 name=None):
        super(ConvBNBlock, self).__init__()
        self._in_c = in_c
        self._out_c = out_c
        self._filter_size = filter_size
        self._dillation = dillation
        self._block_sie = block_size
        self.convs = []
        for n in range(block_size):
            name = None if name is None else name + "_" + str(n)
            conv = ConvBNRelu(
                in_c=self._in_c,
                out_c=self._out_c,
                filter_size=self._filter_size,
                dilation=self._dillation,
                is_test=False,
                use_cudnn=True,
                name=name)
            self.convs.append(conv)

    def forward(self, input):
        tmp = input
        for conv in self.convs:
            tmp = conv(input)
        return tmp


class ConvBNEncoderLayer(Layer):
    def __init__(self,
                 filters=[3, 5, 7],
                 dillations=[1, 1, 1],
                 hidden_size=768,
                 name="encoder"):
        super(ConvBNEncoderLayer, self).__init__()
        cells = []
        self._hidden_size = hidden_size

        self.conv0 = ConvBNRelu(
            in_c=1,
            out_c=self._hidden_size,
            filter_size=[3, self._hidden_size],
            dilation=1,
            is_test=False,
            use_cudnn=True,
            name="sten")

        self.blocks = []
        n = 0
        for filter_n, dillation_n in zip(filters, dillations):
            name = None if name is None else name + "_block" + str(n)
            block = ConvBNBlock(
                self._hidden_size,
                self._hidden_size,
                filter_size=[filter_n, 1],
                dillation=dillation_n,
                block_size=4,
                name=name)
            self.blocks.append(block)
            n += 1

    def forward(self, enc_input):
        tmp = fluid.layers.reshape(
            enc_input, [-1, 1, enc_input.shape[1],
                        self._hidden_size])  #(bs, 1, seq_len, hidden_size)

        tmp = self.conv0(tmp)  # (bs, hidden_size, seq_len, 1)
        for block in self.blocks:
            tmp = block(tmp)

        return tmp
