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

GConv_PRIMITIVES = [
    'std_gconv_3', 'std_gconv_5', 'std_gconv_7', 'dil_gconv_3', 'dil_gconv_5',
    'dil_gconv_7', 'avg_pool_3', 'max_pool_3', 'none', 'skip_connect'
]

ConvBN_PRIMITIVES = [
    'std_conv_bn_3', 'std_conv_bn_5', 'std_conv_bn_7', 'dil_conv_bn_3',
    'dil_conv_bn_5', 'dil_conv_bn_7', 'avg_pool_3', 'max_pool_3', 'none',
    'skip_connect'
]

channel = 768
input_size = 128 * 1

FLOPs = {
    'std_conv_bn_3': input_size * (channel**2) * 3,
    'std_conv_bn_5': input_size * (channel**2) * 5,
    'std_conv_bn_7': input_size * (channel**2) * 7,
    'dil_conv_bn_3': input_size * (channel**2) * 3,
    'dil_conv_bn_5': input_size * (channel**2) * 5,
    'dil_conv_bn_7': input_size * (channel**2) * 7,
    'std_gconv_3': input_size * (channel**2) * 3,
    'std_gconv_5': input_size * (channel**2) * 5,
    'std_gconv_7': input_size * (channel**2) * 7,
    'dil_gconv_3': input_size * (channel**2) * 3,
    'dil_gconv_5': input_size * (channel**2) * 5,
    'dil_gconv_7': input_size * (channel**2) * 7,
    'avg_pool_3': input_size * channel * 3 * 1,
    'max_pool_3': input_size * channel * 3 * 1,
    'none': 0,
    'skip_connect': 0,
}

ModelSize = {
    'std_conv_bn_3': (channel**2) * 3 * 1,
    'std_conv_bn_5': (channel**2) * 5 * 1,
    'std_conv_bn_7': (channel**2) * 7 * 1,
    'dil_conv_bn_3': (channel**2) * 3 * 1,
    'dil_conv_bn_5': (channel**2) * 5 * 1,
    'dil_conv_bn_7': (channel**2) * 7 * 1,
    'std_gconv_3': (channel**2) * 3 * 1,
    'std_gconv_5': (channel**2) * 5 * 1,
    'std_gconv_7': (channel**2) * 7 * 1,
    'dil_gconv_3': (channel**2) * 3 * 1,
    'dil_gconv_5': (channel**2) * 5 * 1,
    'dil_gconv_7': (channel**2) * 7 * 1,
    'avg_pool_3': 0,
    'max_pool_3': 0,
    'none': 0,
    'skip_connect': 0,
}


OPS = {
    'std_gconv_3': lambda n_channel, name: GateConv(n_channel, n_channel, filter_size=[3, 1], dilation=1, name=name),
    'std_gconv_5': lambda n_channel, name: GateConv(n_channel, n_channel, filter_size=[5, 1], dilation=1, name=name),
    'std_gconv_7': lambda n_channel, name: GateConv(n_channel, n_channel, filter_size=[7, 1], dilation=1, name=name),
    'dil_gconv_3': lambda n_channel, name: GateConv(n_channel, n_channel, filter_size=[3, 1], dilation=2, name=name),
    'dil_gconv_5': lambda n_channel, name: GateConv(n_channel, n_channel, filter_size=[5, 1], dilation=2, name=name),
    'dil_gconv_7': lambda n_channel, name: GateConv(n_channel, n_channel, filter_size=[7, 1], dilation=2, name=name),
    'std_conv_bn_3': lambda n_channel, name: ConvBNRelu(n_channel, n_channel, filter_size=[3, 1], dilation=1, name=name),
    'std_conv_bn_5': lambda n_channel, name: ConvBNRelu(n_channel, n_channel, filter_size=[5, 1], dilation=1, name=name),
    'std_conv_bn_7': lambda n_channel, name: ConvBNRelu(n_channel, n_channel, filter_size=[7, 1], dilation=1, name=name),
    'dil_conv_bn_3': lambda n_channel, name: ConvBNRelu(n_channel, n_channel, filter_size=[3, 1], dilation=2, name=name),
    'dil_conv_bn_5': lambda n_channel, name: ConvBNRelu(n_channel, n_channel, filter_size=[5, 1], dilation=2, name=name),
    'dil_conv_bn_7': lambda n_channel, name: ConvBNRelu(n_channel, n_channel, filter_size=[7, 1], dilation=2, name=name),

    'avg_pool_3': lambda n_channel, name: Pool2D(pool_size=(3, 1), pool_padding=(1, 0), pool_type='avg'),
    'max_pool_3': lambda n_channel, name: Pool2D(pool_size=(3, 1), pool_padding=(1, 0), pool_type='max'),
    'none': lambda n_channel, name: Zero(),
    'skip_connect': lambda n_channel, name: Identity(),
}


class MixedOp(fluid.dygraph.Layer):
    def __init__(self, n_channel, name=None, conv_type="conv_bn"):
        super(MixedOp, self).__init__()
        if conv_type == "conv_bn":
            PRIMITIVES = ConvBN_PRIMITIVES
        elif conv_type == "gconv":
            PRIMITIVES = GConv_PRIMITIVES
        ops = [
            OPS[primitive](n_channel, name
                           if name is None else name + "/" + primitive)
            for primitive in PRIMITIVES
        ]
        self._ops = fluid.dygraph.LayerList(ops)
        self.max_flops = max([FLOPs[primitive] for primitive in PRIMITIVES])
        self.max_model_size = max(
            [ModelSize[primitive] for primitive in PRIMITIVES])

    def forward(self, x, weights, flops=[], model_size=[]):
        for i in range(len(self._ops)):
            if weights[i].numpy() != 0:
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
        return bn


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


class Cell(fluid.dygraph.Layer):
    def __init__(self, steps, n_channel, name=None, conv_type="conv_bn"):
        super(Cell, self).__init__()
        self._steps = steps

        self.max_flops = 0
        self.max_model_size = 0
        ops = []
        for i in range(self._steps):
            for j in range(2 + i):
                op = MixedOp(
                    n_channel,
                    name=name
                    if name is None else "%s/step%d_edge%d" % (name, i, j),
                    conv_type=conv_type)
                self.max_flops += op.max_flops
                self.max_model_size += op.max_model_size
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

    def __init__(self,
                 n_layer,
                 hidden_size=768,
                 name="encoder",
                 conv_type="conv_bn",
                 search_layer=True):
        super(EncoderLayer, self).__init__()
        cells = []
        self._n_layer = n_layer
        self._hidden_size = hidden_size
        self._steps = 3
        self._search_layer = search_layer
        self.max_flops = 0
        self.max_model_size = 0
        if conv_type == "conv_bn":
            self._n_ops = len(ConvBN_PRIMITIVES)
            self.conv0 = ConvBNRelu(
                in_c=1,
                out_c=self._hidden_size,
                filter_size=[3, self._hidden_size],
                dilation=1,
                is_test=False,
                use_cudnn=True,
                name="conv0")

        elif conv_type == "gconv":
            self._n_ops = len(GConv_PRIMITIVES)
            self.conv0 = GateConv(
                in_c=1,
                out_c=self._hidden_size,
                filter_size=[3, self._hidden_size],
                dilation=1,
                is_test=False,
                use_cudnn=True,
                name="conv0")

        cells = []
        for i in range(n_layer):
            cell = Cell(
                steps=self._steps,
                n_channel=self._hidden_size,
                name="%s/layer_%d" % (name, i),
                conv_type=conv_type)
            cells.append(cell)
            self.max_flops += cell.max_flops
            self.max_model_size += cell.max_model_size

        self._cells = fluid.dygraph.LayerList(cells)

        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = self._n_ops
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
        tmp = fluid.layers.reshape(
            enc_input, [-1, 1, enc_input.shape[1],
                        self._hidden_size])  #(bs, 1, seq_len, hidden_size)

        tmp = self.conv0(tmp)  # (bs, hidden_size, seq_len, 1)

        alphas = gumbel_softmax(self.alphas)
        k = fluid.layers.reshape(gumbel_softmax(self.k), [-1])

        outputs = []
        s0 = s1 = tmp
        for i in range(self._n_layer):
            s0, s1 = s1, self._cells[i](
                s0, s1, alphas, flops=flops,
                model_size=model_size)  # (bs, hidden_size, seq_len, 1)
            enc_output = fluid.layers.transpose(
                s1, [0, 2, 1, 3])  # (bs, seq_len, hidden_size, 1)
            enc_output = fluid.layers.reshape(
                enc_output, [-1, enc_output.shape[1],
                             self._hidden_size])  # (bs, seq_len, hidden_size)
            outputs.append(enc_output)
            if self._search_layer and k[i].numpy() != 0:
                outputs[-1] = outputs[-1] * k[i]
                return outputs, k[i]
        return outputs, 1.0
