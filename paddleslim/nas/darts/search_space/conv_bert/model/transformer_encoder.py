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

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Embedding, LayerNorm, Linear, Layer, Conv2D, BatchNorm, Pool2D, to_variable
from paddle.fluid.initializer import NormalInitializer

PRIMITIVES = [
    'std_conv_3', 'std_conv_5', 'std_conv_7', 'dil_conv_3', 'dil_conv_5',
    'dil_conv_7', 'avg_pool_3', 'max_pool_3', 'none', 'skip_connect'
]

input_size = 128 * 768

FLOPs = {
    'std_conv_3': input_size * 3 * 1,
    'std_conv_5': input_size * 5 * 1,
    'std_conv_7': input_size * 7 * 1,
    'dil_conv_3': input_size * 3 * 1,
    'dil_conv_5': input_size * 5 * 1,
    'dil_conv_7': input_size * 7 * 1,
    'avg_pool_3': input_size * 3 * 1,
    'max_pool_3': input_size * 3 * 1,
    'none': 0,
    'skip_connect': 0,
}

ModelSize = {
    'std_conv_3': 3 * 1,
    'std_conv_5': 5 * 1,
    'std_conv_7': 7 * 1,
    'dil_conv_3': 3 * 1,
    'dil_conv_5': 5 * 1,
    'dil_conv_7': 7 * 1,
    'avg_pool_3': 0,
    'max_pool_3': 0,
    'none': 0,
    'skip_connect': 0,
}

OPS = {
    'std_conv_3': lambda : ConvBN(1, 1, filter_size=3, dilation=1),
    'std_conv_5': lambda : ConvBN(1, 1, filter_size=5, dilation=1),
    'std_conv_7': lambda : ConvBN(1, 1, filter_size=7, dilation=1),
    'dil_conv_3': lambda : ConvBN(1, 1, filter_size=3, dilation=2),
    'dil_conv_5': lambda : ConvBN(1, 1, filter_size=5, dilation=2),
    'dil_conv_7': lambda : ConvBN(1, 1, filter_size=7, dilation=2),
    'avg_pool_3': lambda : Pool2D(pool_size=(3, 1), pool_type='avg'),
    'max_pool_3': lambda : Pool2D(pool_size=(3, 1), pool_type='max'),
    'none': lambda : Zero(),
    'skip_connect': lambda : Identity(),
}


class MixedOp(fluid.dygraph.Layer):
    def __init__(self):
        super(MixedOp, self).__init__()
        ops = [OPS[primitive]() for primitive in PRIMITIVES]
        self._ops = fluid.dygraph.LayerList(ops)

    def forward(self, x, weights, flops=[], model_size=[]):
        for i in range(len(self._ops)):
            if weights[i] != 0:
                flops.append(FLOPs.values()[i] * weights[i])
                model_size.append(ModelSize.values()[i] * weights[i])
                return self._ops[i](x) * weights[i]


class Zero(fluid.dygraph.Layer):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        x = fluid.layers.zeros_like(x)
        return x


class Identity(fluid.dygraph.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def gumbel_softmax(logits, temperature=0.1, hard=True, eps=1e-20):

    U = np.random.uniform(0, 1, logits.shape)
    logits = logits - to_variable(
        np.log(-np.log(U + eps) + eps).astype("float32"))
    logits = logits / temperature
    logits = fluid.layers.softmax(logits)
    if hard:
        maxes = fluid.layers.reduce_max(logits, dim=1, keep_dim=True)
        hard = fluid.layers.cast((logits == maxes), logits.dtype)
        tmp = hard - logits
        tmp.stop_gradient = True
        out = tmp + logits
    return out


class ConvBN(fluid.dygraph.Layer):
    def __init__(self,
                 out_ch,
                 in_ch,
                 filter_size=3,
                 dilation=1,
                 act="relu",
                 is_test=False,
                 use_cudnn=True):
        super(ConvBN, self).__init__()
        conv_std = (2.0 / (filter_size**2 * in_ch))**0.5
        conv_param = fluid.ParamAttr(
            initializer=fluid.initializer.Normal(0.0, conv_std))

        self.conv_layer = Conv2D(
            in_ch,
            out_ch, [filter_size, 1],
            dilation=dilation,
            padding=[(filter_size - 1) // 2, 0],
            param_attr=conv_param,
            bias_attr=False,
            act=None,
            use_cudnn=use_cudnn)
        self.bn_layer = BatchNorm(out_ch, act=act, is_test=is_test)

    def forward(self, inputs):
        conv = self.conv_layer(inputs)
        bn = self.bn_layer(conv)
        return bn


class Cell(fluid.dygraph.Layer):
    def __init__(self, steps):
        super(Cell, self).__init__()
        self._steps = steps

        ops = []
        for i in range(self._steps):
            for j in range(2 + i):
                op = MixedOp()
                ops.append(op)
        self._ops = fluid.dygraph.LayerList(ops)

    def forward(self, s0, s1, weights, weights2=None, flops=[], model_size=[]):

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = fluid.layers.sums([
                self._ops[offset + j](h,
                                      weights[offset + j],
                                      flops=flops,
                                      model_size=model_size)
                for j, h in enumerate(states)
            ])
            offset += len(states)
            states.append(s)
        out = fluid.layers.sum(states[-self._steps:])
        return out


class EncoderLayer(Layer):
    """
    encoder
    """

    def __init__(self, n_layer, d_model=128, name=""):

        super(EncoderLayer, self).__init__()
        cells = []
        self._n_layer = n_layer
        self._d_model = d_model
        self._steps = 3

        cells = []
        for i in range(n_layer):
            cells.append(Cell(steps=self._steps))
        self._cells = fluid.dygraph.LayerList(cells)

        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        self.alphas = fluid.layers.create_parameter(
            shape=[k, num_ops],
            dtype="float32",
            default_initializer=NormalInitializer(
                loc=0.0, scale=1e-3))

        self.k = fluid.layers.create_parameter(
            shape=[1, self._n_layer],
            dtype="float32",
            default_initializer=NormalInitializer(
                loc=0.0, scale=1e-3))

    def forward(self, enc_input, flops=[], model_size=[]):
        """
        forward
        :param enc_input:
        :param attn_bias:
        :return:
        """
        tmp = fluid.layers.reshape(enc_input,
                                   [-1, 1, enc_input.shape[1], self._d_model])

        alphas = gumbel_softmax(self.alphas)
        k = gumbel_softmax(self.k)

        outputs = []
        s0 = s1 = tmp
        for i in range(self._n_layer):
            s0, s1 = s1, self._cells[i](s0,
                                        s1,
                                        alphas,
                                        flops=flops,
                                        model_size=model_size)
            enc_output = fluid.layers.reshape(
                s1, [-1, enc_input.shape[1], self._d_model])
            outputs.append(enc_output)
            if k[i] != 0:
                outputs[-1] = outputs[-1] * k[i]
                break
        return outputs, k[i]
