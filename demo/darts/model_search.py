# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle.nn.initializer import Normal, KaimingUniform, Constant
from paddle.nn import Conv2D, Pool2D, BatchNorm, Linear
from genotypes import PRIMITIVES
from operations import *
import paddleslim


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups

    # reshape
    x = paddle.reshape(x,
                       [batchsize, groups, channels_per_group, height, width])
    x = paddle.transpose(x, [0, 2, 1, 3, 4])

    # flatten
    x = paddle.reshape(x, [batchsize, num_channels, height, width])
    return x


class MixedOp(paddle.nn.Layer):
    def __init__(self, c_cur, stride, method):
        super(MixedOp, self).__init__()
        self._method = method
        self._k = 1
        self.mp = Pool2D(
            pool_size=2,
            pool_stride=2,
            pool_type='max', )
        ops = []
        for primitive in PRIMITIVES:
            op = OPS[primitive](c_cur // self._k, stride, False)
            if 'pool' in primitive:
                gama = paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=1),
                    trainable=False)
                beta = paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0),
                    trainable=False)
                BN = BatchNorm(
                    c_cur // self._k, param_attr=gama, bias_attr=beta)
                op = paddle.nn.Sequential(op, BN)
            ops.append(op)
        self._ops = paddle.nn.LayerList(ops)

    def forward(self, x, weights):
        return paddle.add_n(
            [weights[i] * op(x) for i, op in enumerate(self._ops)])


class Cell(paddle.nn.Layer):
    def __init__(self, steps, multiplier, c_prev_prev, c_prev, c_cur, reduction,
                 reduction_prev, method):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, c_cur, False)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, c_cur, 1, 1, 0, False)
        self.preprocess1 = ReLUConvBN(c_prev, c_cur, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self._method = method

        ops = []
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(c_cur, stride, method)
                ops.append(op)
        self._ops = paddle.nn.LayerList(ops)

    def forward(self, s0, s1, weights, weights2=None):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = paddle.add_n([
                self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            ])
            offset += len(states)
            states.append(s)
        out = paddle.concat(states[-self._multiplier:], axis=1)
        return out


class Network(paddle.nn.Layer):
    def __init__(self,
                 c_in,
                 num_classes,
                 layers,
                 method,
                 steps=4,
                 multiplier=4,
                 stem_multiplier=3):
        super(Network, self).__init__()
        self._c_in = c_in
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self._primitives = PRIMITIVES
        self._method = method

        c_cur = stem_multiplier * c_in
        self.stem = paddle.nn.Sequential(
            Conv2D(
                num_channels=3,
                num_filters=c_cur,
                filter_size=3,
                padding=1,
                param_attr=paddle.ParamAttr(initializer=KaimingUniform()),
                bias_attr=False),
            BatchNorm(
                num_channels=c_cur,
                param_attr=paddle.ParamAttr(initializer=Constant(value=1)),
                bias_attr=paddle.ParamAttr(initializer=Constant(value=0))))

        c_prev_prev, c_prev, c_cur = c_cur, c_cur, c_in
        cells = []
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                c_cur *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, c_prev_prev, c_prev, c_cur,
                        reduction, reduction_prev, method)
            reduction_prev = reduction
            cells.append(cell)
            c_prev_prev, c_prev = c_prev, multiplier * c_cur
        self.cells = paddle.nn.LayerList(cells)
        self.global_pooling = Pool2D(pool_type='avg', global_pooling=True)
        self.classifier = Linear(
            c_prev,
            num_classes,
            weight_attr=paddle.ParamAttr(initializer=KaimingUniform()),
            bias_attr=paddle.ParamAttr(initializer=KaimingUniform()))

        self._initialize_alphas()

    def forward(self, input):
        s0 = s1 = self.stem(input)
        weights2 = None
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = paddle.nn.functional.softmax(self.alphas_reduce)
            else:
                weights = paddle.nn.functional.softmax(self.alphas_normal)
            s0, s1 = s1, cell(s0, s1, weights, weights2)
        out = self.global_pooling(s1)
        out = paddle.squeeze(out, axes=[2, 3])
        logits = self.classifier(out)
        return logits

    def _loss(self, input, target):
        logits = self(input)
        loss = paddle.mean(
            paddle.nn.functional.softmax_with_cross_entropy(logits, target))
        return loss

    def new(self):
        model_new = Network(self._c_in, self._num_classes, self._layers,
                            self._method)
        return model_new

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(self._primitives)
        self.alphas_normal = paddle.static.create_parameter(
            shape=[k, num_ops],
            dtype="float32",
            default_initializer=Normal(
                loc=0.0, scale=1e-3))
        self.alphas_reduce = paddle.static.create_parameter(
            shape=[k, num_ops],
            dtype="float32",
            default_initializer=Normal(
                loc=0.0, scale=1e-3))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters
