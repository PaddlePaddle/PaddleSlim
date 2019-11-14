# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from .search_space_base import SearchSpaceBase
from .base_layer import conv_bn_layer
from .search_space_registry import SEARCHSPACE

__all__ = ["ResNetSpace"]


@SEARCHSPACE.register
class ResNetSpace(SearchSpaceBase):
    def __init__(self,
                 input_size,
                 output_size,
                 block_num,
                 extract_feature=False,
                 class_dim=1000):
        super(ResNetSpace, self).__init__(input_size, output_size, block_num)
        self.filter_num1 = np.array([48, 64, 96, 128, 160, 192, 224])  #7 
        self.filter_num2 = np.array([64, 96, 128, 160, 192, 256, 320])  #7
        self.filter_num3 = np.array([128, 160, 192, 256, 320, 384])  #6
        self.filter_num4 = np.array([192, 256, 384, 512, 640])  #5
        self.repeat1 = [2, 3, 4, 5, 6]  #5
        self.repeat2 = [2, 3, 4, 5, 6, 7]  #6
        self.repeat3 = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24]  #13
        self.repeat4 = [2, 3, 4, 5, 6, 7]  #6
        self.class_dim = class_dim
        self.extract_feature = extract_feature

    def init_tokens(self):
        init_token_base = [0, 0, 0, 0, 0, 0, 0, 0]
        self.token_len = self.block_num * 2
        return init_token_base[:self.token_len]

    def range_table(self):
        range_table_base = [3, 3, 3, 3, 3, 3, 3, 3]
        return range_table_base[:self.token_len]

    def token2arch(self, tokens=None):
        assert self.block_num < 5, 'block number must less than 5, but receive block number is {}'.format(
            self.block_num)

        if tokens is None:
            tokens = self.init_tokens()

        def net_arch(input):
            depth = []
            num_filters = []
            if self.block_num >= 1:
                filter1 = self.filter_num1[tokens[0]]
                repeat1 = self.repeat1[tokens[1]]
                depth.append(filter1)
                num_filters.append(repeat1)
            if self.block_num >= 2:
                filter2 = self.filter_num2[tokens[2]]
                repeat2 = self.repeat2[tokens[3]]
                depth.append(filter2)
                num_filters.append(repeat2)
            if self.block_num >= 3:
                filter3 = self.filter_num3[tokens[4]]
                repeat3 = self.repeat3[tokens[5]]
                depth.append(filter3)
                num_filters.append(repeat3)
            if self.block_num >= 4:
                filter4 = self.filter_num4[tokens[6]]
                repeat4 = self.repeat4[tokens[7]]
                depth.append(filter4)
                num_filters.append(repeat4)

            conv = conv_bn_layer(
                input=input,
                filter_size=5,
                num_filters=filter1,
                stride=2,
                act='relu',
                name='resnet_conv0')
            for block in range(len(depth)):
                for i in range(depth[block]):
                    conv = self._basicneck_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        name='resnet_depth{}_block{}'.format(i, block))

            if self.output_size == 1:
                conv = fluid.layers.fc(
                    input=conv,
                    size=self.class_dim,
                    act=None,
                    param_attr=fluid.param_attr.ParamAttr(
                        initializer=fluid.initializer.NormalInitializer(0.0,
                                                                        0.01)),
                    bias_attr=fluid.param_attr.ParamAttr(
                        initializer=fluid.initializer.ConstantInitializer(0)))

            return conv

        return net_arch

    def _shortcut(self, input, ch_out, stride, name=None):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return conv_bn_layer(
                input=input,
                filter_size=1,
                num_filters=ch_out,
                stride=stride,
                name=name + '_conv')
        else:
            return input

    def _basicneck_block(self, input, num_filters, stride, name=None):
        conv0 = conv_bn_layer(
            input=input,
            filter_size=3,
            num_filters=num_filters,
            stride=stride,
            act='relu',
            name=name + '_basicneck_conv0')
        conv1 = conv_bn_layer(
            input=conv0,
            filter_size=3,
            num_filters=num_filters,
            stride=1,
            act=None,
            name=name + '_basicneck_conv1')
        short = self._shortcut(
            input, num_filters, stride, name=name + '_short')
        return fluid.layers.elementwise_add(
            x=short, y=conv1, act='relu', name=name + '_basicneck_add')
