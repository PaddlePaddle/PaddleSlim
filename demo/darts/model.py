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

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import ConstantInitializer, MSRAInitializer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable
from genotypes import PRIMITIVES
from genotypes import Genotype
from operations import *


class ConvBN(fluid.dygraph.Layer):
    def __init__(self, c_curr, c_out, kernel_size, padding, stride, name=None):
        super(ConvBN, self).__init__()
        self.conv = Conv2D(
            num_channels=c_curr,
            num_filters=c_out,
            filter_size=kernel_size,
            stride=stride,
            padding=padding,
            param_attr=fluid.ParamAttr(
                name=name + "_conv" if name is not None else None,
                initializer=MSRAInitializer()),
            bias_attr=False)
        self.bn = BatchNorm(
            num_channels=c_out,
            param_attr=fluid.ParamAttr(
                name=name + "_bn_scale" if name is not None else None,
                initializer=ConstantInitializer(value=1)),
            bias_attr=fluid.ParamAttr(
                name=name + "_bn_offset" if name is not None else None,
                initializer=ConstantInitializer(value=0)),
            moving_mean_name=name + "_bn_mean" if name is not None else None,
            moving_variance_name=name + "_bn_variance"
            if name is not None else None)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        return bn


class Classifier(fluid.dygraph.Layer):
    def __init__(self, input_dim, num_classes, name=None):
        super(Classifier, self).__init__()
        self.pool2d = Pool2D(pool_type='avg', global_pooling=True)
        self.fc = Linear(
            input_dim=input_dim,
            output_dim=num_classes,
            param_attr=fluid.ParamAttr(
                name=name + "_fc_weights" if name is not None else None,
                initializer=MSRAInitializer()),
            bias_attr=fluid.ParamAttr(
                name=name + "_fc_bias" if name is not None else None,
                initializer=MSRAInitializer()))

    def forward(self, x):
        x = self.pool2d(x)
        x = fluid.layers.squeeze(x, axes=[2, 3])
        out = self.fc(x)
        return out


def drop_path(x, drop_prob):
    if drop_prob > 0:
        keep_prob = 1. - drop_prob
    mask = 1 - np.random.binomial(
        1, drop_prob, size=[x.shape[0]]).astype(np.float32)
    mask = to_variable(mask)
    x = fluid.layers.elementwise_mul(x / keep_prob, mask, axis=0)
    return x


class Cell(fluid.dygraph.Layer):
    def __init__(self, genotype, c_prev_prev, c_prev, c_curr, reduction,
                 reduction_prev):
        super(Cell, self).__init__()
        print(c_prev_prev, c_prev, c_curr)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, c_curr)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, c_curr, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(c_prev, c_curr, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        multiplier = len(concat)
        self._multiplier = multiplier
        self._compile(c_curr, op_names, indices, multiplier, reduction)

    def _compile(self, c_curr, op_names, indices, multiplier, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        ops = []
        edge_index = 0
        for op_name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[op_name](c_curr, stride, True)
            ops += [op]
            edge_index += 1
        self._ops = fluid.dygraph.LayerList(ops)
        self._indices = indices

    def forward(self, s0, s1, drop_prob, training):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            states += [h1 + h2]
        out = fluid.layers.concat(input=states[-self._multiplier:], axis=1)
        return out


class AuxiliaryHeadCIFAR(fluid.dygraph.Layer):
    def __init__(self, C, num_classes):
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.avgpool = Pool2D(
            pool_size=5, pool_stride=3, pool_padding=0, pool_type='avg')
        self.conv_bn1 = ConvBN(
            c_curr=C,
            c_out=128,
            kernel_size=1,
            padding=0,
            stride=1,
            name='aux_conv_bn1')
        self.conv_bn2 = ConvBN(
            c_curr=128,
            c_out=768,
            kernel_size=2,
            padding=0,
            stride=1,
            name='aux_conv_bn2')
        self.classifier = Classifier(768, num_classes, 'aux')

    def forward(self, x):
        x = fluid.layers.relu(x)
        x = self.avgpool(x)
        conv1 = self.conv_bn1(x)
        conv1 = fluid.layers.relu(conv1)
        conv2 = self.conv_bn2(conv1)
        conv2 = fluid.layers.relu(conv2)
        out = self.classifier(conv2)
        return out


class NetworkCIFAR(fluid.dygraph.Layer):
    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        c_curr = stem_multiplier * C
        self.stem = ConvBN(
            c_curr=3, c_out=c_curr, kernel_size=3, padding=1, stride=1)

        c_prev_prev, c_prev, c_curr = c_curr, c_curr, C
        cells = []
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                c_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, c_prev_prev, c_prev, c_curr, reduction,
                        reduction_prev)
            reduction_prev = reduction
            cells += [cell]
            c_prev_prev, c_prev = c_prev, cell._multiplier * c_curr
            if i == 2 * layers // 3:
                c_to_auxiliary = c_prev
        self.cells = fluid.dygraph.LayerList(cells)

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(c_to_auxiliary,
                                                     num_classes)
        self.classifier = Classifier(c_prev, num_classes)

    def forward(self, input, drop_path_prob, training):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, drop_path_prob, training)
            if i == 2 * self._layers // 3:
                if self._auxiliary and training:
                    logits_aux = self.auxiliary_head(s1)
        logits = self.classifier(s1)
        return logits, logits_aux


class AuxiliaryHeadImageNet(fluid.dygraph.Layer):
    def __init__(self, C, num_classes):
        super(AuxiliaryHeadImageNet, self).__init__()
        self.avgpool = Pool2D(
            pool_size=5, pool_stride=2, pool_padding=0, pool_type='avg')
        self.conv_bn1 = ConvBN(
            c_curr=C,
            c_out=128,
            kernel_size=1,
            padding=0,
            stride=1,
            name='aux_conv_bn1')
        self.conv_bn2 = ConvBN(
            c_curr=128,
            c_out=768,
            kernel_size=2,
            padding=0,
            stride=1,
            name='aux_conv_bn2')
        self.classifier = Classifier(768, num_classes, 'aux')

    def forward(self, x):
        x = fluid.layers.relu(x)
        x = self.avgpool(x)
        conv1 = self.conv_bn1(x)
        conv1 = fluid.layers.relu(conv1)
        conv2 = self.conv_bn2(conv1)
        conv2 = fluid.layers.relu(conv2)
        out = self.classifier(conv2)
        return out


class NetworkImageNet(fluid.dygraph.Layer):
    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem_a0 = ConvBN(
            c_curr=3, c_out=C // 2, kernel_size=3, padding=1, stride=2)

        self.stem_a1 = ConvBN(
            c_curr=C // 2, c_out=C, kernel_size=3, padding=1, stride=2)

        self.stem_b = ConvBN(
            c_curr=C, c_out=C, kernel_size=3, padding=1, stride=2)

        c_prev_prev, c_prev, c_curr = C, C, C
        cells = []
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                c_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, c_prev_prev, c_prev, c_curr, reduction,
                        reduction_prev)
            reduction_prev = reduction
            cells += [cell]
            c_prev_prev, c_prev = c_prev, cell._multiplier * c_curr
            if i == 2 * layers // 3:
                c_to_auxiliary = c_prev
        self.cells = fluid.dygraph.LayerList(cells)

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(c_to_auxiliary,
                                                        num_classes)
        self.classifier = Classifier(c_prev, num_classes)

    def forward(self, input, training):
        logits_aux = None
        s0 = self.stem_a0(input)
        s0 = fluid.layers.relu(s0)
        s0 = self.stem_a1(s0)
        s1 = fluid.layers.relu(s0)
        s1 = self.stem_b(s1)

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, 0, training)
            if i == 2 * self._layers // 3:
                if self._auxiliary and training:
                    logits_aux = self.auxiliary_head(s1)
        logits = self.classifier(s1)
        return logits, logits_aux
