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
from paddle.fluid import ParamAttr
from paddle.fluid.initializer import MSRA, ConstantInitializer

ConvBN_PRIMITIVES = [
    'std_conv_bn_3', 'std_conv_bn_5', 'std_conv_bn_7', 'dil_conv_bn_3',
    'dil_conv_bn_5', 'dil_conv_bn_7', 'avg_pool_3', 'max_pool_3', 'none',
    'skip_connect'
]


OPS = {
    'std_conv_bn_3': lambda n_channel, name: ReluConvBN(n_channel, n_channel, filter_size=[3, 1], dilation=1, name=name),
    'std_conv_bn_5': lambda n_channel, name: ReluConvBN(n_channel, n_channel, filter_size=[5, 1], dilation=1, name=name),
    'std_conv_bn_7': lambda n_channel, name: ReluConvBN(n_channel, n_channel, filter_size=[7, 1], dilation=1, name=name),
    'dil_conv_bn_3': lambda n_channel, name: ReluConvBN(n_channel, n_channel, filter_size=[3, 1], dilation=2, name=name),
    'dil_conv_bn_5': lambda n_channel, name: ReluConvBN(n_channel, n_channel, filter_size=[5, 1], dilation=2, name=name),
    'dil_conv_bn_7': lambda n_channel, name: ReluConvBN(n_channel, n_channel, filter_size=[7, 1], dilation=2, name=name),

    'avg_pool_3': lambda n_channel, name: Pool2D(pool_size=(3,1), pool_padding=(1, 0), pool_type='avg'),
    'max_pool_3': lambda n_channel, name: Pool2D(pool_size=(3,1), pool_padding=(1, 0), pool_type='max'),
    'none': lambda n_channel, name: Zero(),
    'skip_connect': lambda n_channel, name: Identity(),
}


class MixedOp(fluid.dygraph.Layer):
    def __init__(self, n_channel, name=None):
        super(MixedOp, self).__init__()
        PRIMITIVES = ConvBN_PRIMITIVES
        # ops = [
        #     OPS[primitive](n_channel, name
        #                    if name is None else name + "/" + primitive)
        #     for primitive in PRIMITIVES
        # ]
        ops = []
        for primitive in PRIMITIVES:
            op = OPS[primitive](n_channel, name
                                if name is None else name + "/" + primitive)
            if 'pool' in primitive:
                gama = ParamAttr(
                    initializer=fluid.initializer.Constant(value=1),
                    trainable=False)
                beta = ParamAttr(
                    initializer=fluid.initializer.Constant(value=0),
                    trainable=False)
                BN = BatchNorm(n_channel, param_attr=gama, bias_attr=beta)
                op = fluid.dygraph.Sequential(op, BN)
            ops.append(op)

        self._ops = fluid.dygraph.LayerList(ops)

    def forward(self, x, weights):
        #out = weights[0] * self._ops[0](x)
        # out = fluid.layers.sums(
        #    [weights[i] * op(x) for i, op in enumerate(self._ops)])
        # return out

        for i in range(len(self._ops)):

            if isinstance(weights, Iterable):
                weights_i = weights[i]
            else:
                weights_i = weights[i].numpy()

            if weights_i != 0:
                return self._ops[i](x) * weights[i]


def gumbel_softmax(logits, temperature=1, hard=True, eps=1e-10):
    #U = np.random.uniform(0, 1, logits.shape)
    #U = - to_variable(
    #    np.log(-np.log(U + eps) + eps).astype("float32"))
    U = np.random.gumbel(0, 1, logits.shape).astype("float32")

    logits = logits + to_variable(U)
    logits = logits / temperature
    logits = fluid.layers.softmax(logits)

    if hard:
        maxes = fluid.layers.reduce_max(logits, dim=1, keep_dim=True)
        hard = fluid.layers.cast((logits == maxes), logits.dtype)
        # out = hard - logits.detach() + logits
        tmp = hard - logits
        tmp.stop_gradient = True
        out = tmp + logits
    else:
        out = logits
    return out


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


class ReluConvBN(fluid.dygraph.Layer):
    def __init__(self,
                 in_c=768,
                 out_c=768,
                 filter_size=[3, 1],
                 dilation=1,
                 stride=1,
                 affine=False,
                 use_cudnn=True,
                 name=None):
        super(ReluConvBN, self).__init__()
        #conv_std = (2.0 /
        #            (filter_size[0] * filter_size[1] * out_c * in_c))**0.5
        conv_param = fluid.ParamAttr(
            name=name if name is None else (name + "_conv.weights"),
            initializer=fluid.initializer.MSRA())

        self.conv = Conv2D(
            in_c,
            out_c,
            filter_size,
            dilation=[dilation, 1],
            stride=stride,
            padding=[(filter_size[0] - 1) * dilation // 2, 0],
            param_attr=conv_param,
            act=None,
            bias_attr=False,
            use_cudnn=use_cudnn)

        gama = ParamAttr(
            initializer=fluid.initializer.Constant(value=1), trainable=affine)
        beta = ParamAttr(
            initializer=fluid.initializer.Constant(value=0), trainable=affine)

        self.bn = BatchNorm(out_c, param_attr=gama, bias_attr=beta)

    def forward(self, inputs):
        inputs = fluid.layers.relu(inputs)
        conv = self.conv(inputs)
        bn = self.bn(conv)
        return bn


class Cell(fluid.dygraph.Layer):
    def __init__(self, steps, n_channel, name=None):
        super(Cell, self).__init__()
        self._steps = steps
        self.preprocess0 = ReluConvBN(in_c=n_channel, out_c=n_channel)
        self.preprocess1 = ReluConvBN(in_c=n_channel, out_c=n_channel)

        ops = []
        for i in range(self._steps):
            for j in range(2 + i):
                op = MixedOp(
                    n_channel,
                    name=name
                    if name is None else "%s/step%d_edge%d" % (name, i, j))
                ops.append(op)
        self._ops = fluid.dygraph.LayerList(ops)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = fluid.layers.sums([
                self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            ])
            offset += len(states)
            states.append(s)
        out = fluid.layers.sums(states[-self._steps:])
        #out = fluid.layers.concat(input=states[-self._steps:], axis=1)
        return out


class EncoderLayer(Layer):
    """
    encoder
    """

    def __init__(self,
                 n_layer,
                 hidden_size=768,
                 name="encoder",
                 search_layer=True,
                 use_fixed_gumbel=False,
                 gumbel_alphas=None):
        super(EncoderLayer, self).__init__()
        self._n_layer = n_layer
        self._hidden_size = hidden_size
        self._n_channel = 256
        self._steps = 3
        self._n_ops = len(ConvBN_PRIMITIVES)
        self.use_fixed_gumbel = use_fixed_gumbel

        self.stem = fluid.dygraph.Sequential(
            Conv2D(
                num_channels=1,
                num_filters=self._n_channel,
                filter_size=[3, self._hidden_size],
                padding=[1, 0],
                param_attr=fluid.ParamAttr(initializer=MSRA()),
                bias_attr=False),
            BatchNorm(
                num_channels=self._n_channel,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(value=1)),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(value=0))))

        cells = []
        for i in range(n_layer):
            cell = Cell(
                steps=self._steps,
                n_channel=self._n_channel,
                name="%s/layer_%d" % (name, i))
            cells.append(cell)

        self._cells = fluid.dygraph.LayerList(cells)

        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = self._n_ops
        self.alphas = fluid.layers.create_parameter(
            shape=[k, num_ops],
            dtype="float32",
            default_initializer=NormalInitializer(
                loc=0.0, scale=1e-3))

        # self.k = fluid.layers.create_parameter(
        #     shape=[1, self._n_layer],
        #     dtype="float32",
        #     default_initializer=NormalInitializer(
        #         loc=0.0, scale=1e-3))
        self.pool2d_avg = Pool2D(pool_type='avg', global_pooling=True)
        self.bns = []
        self.outs = []
        for i in range(self._n_layer):

            bn = BatchNorm(
                num_channels=self._n_channel,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(value=1),
                    trainable=False),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(value=0),
                    trainable=False))
            self.bns.append(bn)

            out = Linear(
                self._n_channel,
                3,
                param_attr=ParamAttr(initializer=MSRA()),
                bias_attr=ParamAttr(initializer=MSRA()))
            self.outs.append(out)

        self.use_fixed_gumbel = use_fixed_gumbel
        self.gumbel_alphas = gumbel_softmax(self.alphas)
        if gumbel_alphas is not None:
            self.gumbel_alphas = np.array(gumbel_alphas).reshape(
                self.alphas.shape)
        else:
            self.gumbel_alphas = gumbel_softmax(self.alphas)
            self.gumbel_alphas.stop_gradient = True

        print("gumbel_alphas: {}".format(self.gumbel_alphas))

    def forward(self, enc_input_0, enc_input_1, flops=[], model_size=[]):
        alphas = self.gumbel_alphas if self.use_fixed_gumbel else gumbel_softmax(
            self.alphas)

        s0 = fluid.layers.reshape(
            enc_input_0, [-1, 1, enc_input_0.shape[1], enc_input_0.shape[2]])
        s1 = fluid.layers.reshape(
            enc_input_1, [-1, 1, enc_input_1.shape[1], enc_input_1.shape[2]])
        # (bs, 1, seq_len, hidden_size)

        s0 = self.stem(s0)
        s1 = self.stem(s1)
        # (bs, n_channel, seq_len, 1)
        if self.use_fixed_gumbel:
            alphas = self.gumbel_alphas
        else:
            alphas = gumbel_softmax(self.alphas)

        outputs = []
        for i in range(self._n_layer):
            s0, s1 = s1, self._cells[i](s0, s1, alphas)
            tmp = self.bns[i](s1)
            tmp = self.pool2d_avg(tmp)
            # (bs, n_channel, seq_len, 1)
            tmp = fluid.layers.reshape(tmp, shape=[-1, 0])
            tmp = self.outs[i](tmp)
            outputs.append(tmp)
        return outputs
