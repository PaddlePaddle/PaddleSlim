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
from collections.abc import Iterable

import paddle

ConvBN_PRIMITIVES = [
    'std_conv_bn_3',
    'std_conv_bn_5',
    'std_conv_bn_7',
    'dil_conv_bn_3',
    'dil_conv_bn_5',
    'dil_conv_bn_7',
    'avg_pool_3',
    'max_pool_3',
    'skip_connect',
    'none',
]


OPS = {
    'std_conv_bn_3': lambda n_channel, name: ReluConvBN(n_channel, n_channel, filter_size=[3, 1], dilation=1, name=name),
    'std_conv_bn_5': lambda n_channel, name: ReluConvBN(n_channel, n_channel, filter_size=[5, 1], dilation=1, name=name),
    'std_conv_bn_7': lambda n_channel, name: ReluConvBN(n_channel, n_channel, filter_size=[7, 1], dilation=1, name=name),
    'dil_conv_bn_3': lambda n_channel, name: ReluConvBN(n_channel, n_channel, filter_size=[3, 1], dilation=2, name=name),
    'dil_conv_bn_5': lambda n_channel, name: ReluConvBN(n_channel, n_channel, filter_size=[5, 1], dilation=2, name=name),
    'dil_conv_bn_7': lambda n_channel, name: ReluConvBN(n_channel, n_channel, filter_size=[7, 1], dilation=2, name=name),

    'avg_pool_3': lambda n_channel, name: paddle.fluid.dygraph.Pool2D(pool_size=(3,1), pool_padding=(1, 0), pool_type='avg'),
    'max_pool_3': lambda n_channel, name: paddle.fluid.dygraph.Pool2D(pool_size=(3,1), pool_padding=(1, 0), pool_type='max'),
    'none': lambda n_channel, name: Zero(),
    'skip_connect': lambda n_channel, name: Identity(),
}


class MixedOp(paddle.nn.Layer):
    def __init__(self, n_channel, name=None):
        super(MixedOp, self).__init__()
        PRIMITIVES = ConvBN_PRIMITIVES
        ops = []
        for primitive in PRIMITIVES:
            op = OPS[primitive](n_channel, name
                                if name is None else name + "/" + primitive)
            if 'pool' in primitive:
                gama = ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=1),
                    trainable=False)
                beta = ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0),
                    trainable=False)
                BN = paddle.nn.BatchNorm(
                    n_channel, param_attr=gama, bias_attr=beta)
                op = paddle.nn.Sequential(op, BN)
            ops.append(op)

        self._ops = paddle.nn.LayerList(ops)

    def forward(self, x, weights):
        # out = fluid.layers.sums(
        #     [weights[i] * op(x) for i, op in enumerate(self._ops)])
        # return out

        for i in range(len(weights.numpy())):
            if weights[i].numpy() != 0:
                return self._ops[i](x) * weights[i]


def gumbel_softmax(logits, epoch, temperature=1.0, hard=True, eps=1e-10):
    temperature = temperature * (0.98**epoch)
    U = np.random.gumbel(0, 1, logits.shape).astype("float32")

    logits = logits + paddle.to_tensor(data=U)
    logits = logits / temperature
    logits = paddle.nn.functional.softmax(logits)

    if hard:
        maxes = paddle.max(x=logits, axis=1, keepdim=True)
        hard = paddle.cast((logits == maxes), logits.dtype)
        out = hard - logits.detach() + logits
        # tmp.stop_gradient = True
        # out = tmp + logits
    else:
        out = logits

    return out


class Zero(paddle.nn.Layer):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        x = paddle.zeros_like(x=x)
        return x


class Identity(paddle.nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ReluConvBN(paddle.nn.Layer):
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
        conv_param = paddle.ParamAttr(
            name=name if name is None else (name + "_conv.weights"),
            initializer=paddle.nn.initializer.KaimingUniform())

        self.conv = paddle.nn.Conv2D(
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
            initializer=paddle.nn.initializer.Constant(value=1),
            trainable=affine)
        beta = ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0),
            trainable=affine)

        self.bn = paddle.nn.BatchNorm(out_c, param_attr=gama, bias_attr=beta)

    def forward(self, inputs):
        inputs = paddle.nn.functional.relu(inputs)
        conv = self.conv(inputs)
        bn = self.bn(conv)
        return bn


class Cell(paddle.nn.Layer):
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
        self._ops = paddle.nn.LayerList(ops)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = paddle.add_n(inputs=[
                self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            ])
            offset += len(states)
            states.append(s)
        out = paddle.add_n(inputs=states[-self._steps:])
        #out = paddle.concat(input=states[-self._steps:], axis=1)
        return out


class EncoderLayer(paddle.nn.Layer):
    """
    encoder
    """

    def __init__(self,
                 num_labels,
                 n_layer,
                 hidden_size=768,
                 name="encoder",
                 search_layer=True,
                 use_fixed_gumbel=False,
                 gumbel_alphas=None):
        super(EncoderLayer, self).__init__()
        self._n_layer = n_layer
        self._hidden_size = hidden_size
        self._n_channel = 128
        self._steps = 3
        self._n_ops = len(ConvBN_PRIMITIVES)
        self.use_fixed_gumbel = use_fixed_gumbel

        self.stem0 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                num_channels=1,
                num_filters=self._n_channel,
                filter_size=[3, self._hidden_size],
                padding=[1, 0],
                param_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.KaimingUniform()),
                bias_attr=False),
            paddle.nn.BatchNorm(
                num_channels=self._n_channel,
                param_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=1)),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0))))

        self.stem1 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                num_channels=1,
                num_filters=self._n_channel,
                filter_size=[3, self._hidden_size],
                padding=[1, 0],
                param_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.KaimingUniform()),
                bias_attr=False),
            paddle.nn.BatchNorm(
                num_channels=self._n_channel,
                param_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=1)),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0))))

        cells = []
        for i in range(n_layer):
            cell = Cell(
                steps=self._steps,
                n_channel=self._n_channel,
                name="%s/layer_%d" % (name, i))
            cells.append(cell)

        self._cells = paddle.nn.LayerList(cells)

        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = self._n_ops
        self.alphas = paddle.static.create_parameter(
            shape=[k, num_ops],
            dtype="float32",
            default_initializer=paddle.nn.initializer.NormalInitializer(
                loc=0.0, scale=1e-3))

        self.pool2d_avg = paddle.fluid.dygraph.Pool2D(
            pool_type='avg', global_pooling=True)
        self.bns = []
        self.outs = []
        for i in range(self._n_layer):
            bn = paddle.nn.BatchNorm(
                num_channels=self._n_channel,
                param_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=1),
                    trainable=False),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0),
                    trainable=False))
            out = paddle.nn.Linear(
                in_features=self._n_channel,
                out_features=num_labels,
                weight_attr=ParamAttr(
                    initializer=paddle.nn.initializer.KaimingUniform()),
                bias_attr=ParamAttr(
                    initializer=paddle.nn.initializer.KaimingUniform()))
            self.bns.append(bn)
            self.outs.append(out)
        self._bns = paddle.nn.LayerList(self.bns)
        self._outs = paddle.nn.LayerList(self.outs)

        self.use_fixed_gumbel = use_fixed_gumbel
        #self.gumbel_alphas = gumbel_softmax(self.alphas, 0).detach()

        mrpc_arch = [
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # std_conv7 0     # node 0
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # dil_conv5 1
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # std_conv7 0     # node 1
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # dil_conv5 1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # zero 2
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # zero 0          # node2
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # std_conv3 1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # zero 2
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # dil_conv3 3
        ]
        self.gumbel_alphas = paddle.to_tensor(
            data=np.array(mrpc_arch).astype(np.float32))
        self.gumbel_alphas.stop_gradient = True
        print("gumbel_alphas: \n", self.gumbel_alphas.numpy())

    def forward(self, enc_input_0, enc_input_1, epoch, flops=[], model_size=[]):
        alphas = self.gumbel_alphas if self.use_fixed_gumbel else gumbel_softmax(
            self.alphas, epoch)

        s0 = paddle.unsqueeze(x=enc_input_0, axis=[1])
        s1 = paddle.unsqueeze(x=enc_input_1, axis=[1])
        s0 = self.stem0(s0)
        s1 = self.stem1(s1)

        enc_outputs = []
        for i in range(self._n_layer):
            s0, s1 = s1, self._cells[i](s0, s1, alphas)
            # (bs, n_channel, seq_len, 1)
            tmp = self._bns[i](s1)
            tmp = self.pool2d_avg(tmp)
            tmp = paddle.reshape(tmp, shape=[-1, 0])
            tmp = self._outs[i](tmp)
            enc_outputs.append(tmp)

        return enc_outputs
