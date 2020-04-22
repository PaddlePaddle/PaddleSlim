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

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import NormalInitializer, MSRAInitializer, ConstantInitializer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable
from genotypes import PRIMITIVES
from operations import *


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups

    # reshape
    x = fluid.layers.reshape(
        x, [batchsize, groups, channels_per_group, height, width])
    x = fluid.layers.transpose(x, [0, 2, 1, 3, 4])

    # flatten
    x = fluid.layers.reshape(x, [batchsize, num_channels, height, width])
    return x


class MixedOp(fluid.dygraph.Layer):
    def __init__(self, c_cur, stride, method):
        super(MixedOp, self).__init__()
        self._method = method
        self._k = 4 if self._method == "PC-DARTS" else 1
        self.mp = Pool2D(
            pool_size=2,
            pool_stride=2,
            pool_type='max', )
        ops = []
        for primitive in PRIMITIVES:
            op = OPS[primitive](c_cur // self._k, stride, False)
            if 'pool' in primitive:
                gama = ParamAttr(
                    initializer=fluid.initializer.Constant(value=1),
                    trainable=False)
                beta = ParamAttr(
                    initializer=fluid.initializer.Constant(value=0),
                    trainable=False)
                BN = BatchNorm(
                    c_cur // self._k, param_attr=gama, bias_attr=beta)
                op = fluid.dygraph.Sequential(op, BN)
            ops.append(op)
        self._ops = fluid.dygraph.LayerList(ops)

    def forward(self, x, weights):
        if self._method == "PC-DARTS":
            dim_2 = x.shape[1]
            xtemp = x[:, :dim_2 // self._k, :, :]
            xtemp2 = x[:, dim_2 // self._k:, :, :]

            temp1 = fluid.layers.sums(
                [weights[i] * op(xtemp) for i, op in enumerate(self._ops)])

            if temp1.shape[2] == x.shape[2]:
                out = fluid.layers.concat([temp1, xtemp2], axis=1)
            else:
                out = fluid.layers.concat([temp1, self.mp(xtemp2)], axis=1)
            out = channel_shuffle(out, self._k)
        else:
            out = fluid.layers.sums(
                [weights[i] * op(x) for i, op in enumerate(self._ops)])
        return out


class Cell(fluid.dygraph.Layer):
    def __init__(self, steps, multiplier, c_prev_prev, c_prev, c_cur,
                 reduction, reduction_prev, method):
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
        self._ops = fluid.dygraph.LayerList(ops)

    def forward(self, s0, s1, weights, weights2=None):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            if self._method == "PC-DARTS":
                s = fluid.layers.sums([
                    weights2[offset + j] *
                    self._ops[offset + j](h, weights[offset + j])
                    for j, h in enumerate(states)
                ])
            else:
                s = fluid.layers.sums([
                    self._ops[offset + j](h, weights[offset + j])
                    for j, h in enumerate(states)
                ])
            offset += len(states)
            states.append(s)
        out = fluid.layers.concat(input=states[-self._multiplier:], axis=1)
        return out


class Network(fluid.dygraph.Layer):
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
        self.stem = fluid.dygraph.Sequential(
            Conv2D(
                num_channels=3,
                num_filters=c_cur,
                filter_size=3,
                padding=1,
                param_attr=fluid.ParamAttr(initializer=MSRAInitializer()),
                bias_attr=False),
            BatchNorm(
                num_channels=c_cur,
                param_attr=fluid.ParamAttr(
                    initializer=ConstantInitializer(value=1)),
                bias_attr=fluid.ParamAttr(
                    initializer=ConstantInitializer(value=0))))

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
        self.cells = fluid.dygraph.LayerList(cells)
        self.global_pooling = Pool2D(pool_type='avg', global_pooling=True)
        self.classifier = Linear(
            input_dim=c_prev,
            output_dim=num_classes,
            param_attr=ParamAttr(initializer=MSRAInitializer()),
            bias_attr=ParamAttr(initializer=MSRAInitializer()))

        self._initialize_alphas()

    def forward(self, input):
        s0 = s1 = self.stem(input)
        weights2 = None
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = fluid.layers.softmax(self.alphas_reduce)
                if self._method == "PC-DARTS":
                    n = 3
                    start = 2
                    weights2 = fluid.layers.softmax(self.betas_reduce[0:2])
                    for i in range(self._steps - 1):
                        end = start + n
                        tw2 = fluid.layers.softmax(self.betas_reduce[start:
                                                                     end])
                        start = end
                        n += 1
                        weights2 = fluid.layers.concat([weights2, tw2])
            else:
                weights = fluid.layers.softmax(self.alphas_normal)
                if self._method == "PC-DARTS":
                    n = 3
                    start = 2
                    weights2 = fluid.layers.softmax(self.betas_normal[0:2])
                    for i in range(self._steps - 1):
                        end = start + n
                        tw2 = fluid.layers.softmax(self.betas_normal[start:
                                                                     end])
                        start = end
                        n += 1
                        weights2 = fluid.layers.concat([weights2, tw2])
            s0, s1 = s1, cell(s0, s1, weights, weights2)
        out = self.global_pooling(s1)
        out = fluid.layers.squeeze(out, axes=[2, 3])
        logits = self.classifier(out)
        return logits

    def _loss(self, input, target):
        logits = self(input)
        loss = fluid.layers.reduce_mean(
            fluid.layers.softmax_with_cross_entropy(logits, target))
        return loss

    def new(self):
        model_new = Network(self._c_in, self._num_classes, self._layers,
                            self._method)
        return model_new

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(self._primitives)
        self.alphas_normal = fluid.layers.create_parameter(
            shape=[k, num_ops],
            dtype="float32",
            default_initializer=NormalInitializer(
                loc=0.0, scale=1e-3))
        self.alphas_reduce = fluid.layers.create_parameter(
            shape=[k, num_ops],
            dtype="float32",
            default_initializer=NormalInitializer(
                loc=0.0, scale=1e-3))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
        if self._method == "PC-DARTS":
            self.betas_normal = fluid.layers.create_parameter(
                shape=[k],
                dtype="float32",
                default_initializer=NormalInitializer(
                    loc=0.0, scale=1e-3))
            self.betas_reduce = fluid.layers.create_parameter(
                shape=[k],
                dtype="float32",
                default_initializer=NormalInitializer(
                    loc=0.0, scale=1e-3))
            self._arch_parameters += [self.betas_normal, self.betas_reduce]

    def arch_parameters(self):
        return self._arch_parameters
