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

__all__ = ["MobileNetV2BlockSpace"]


@SEARCHSPACE.register
class MobileNetV2BlockSpace(SearchSpaceBase):
    def __init__(self,
                 input_size,
                 output_size,
                 block_num,
                 block_mask=None,
                 scale=1.0):
        super(MobileNetV2BlockSpace, self).__init__(input_size, output_size,
                                                    block_num, block_mask)
        self.filter_num1 = np.array([3, 4, 8, 12, 16, 24, 32, 48])
        self.filter_num1 = np.array([3, 4, 8, 12, 16, 24, 32, 48])  #8
        self.filter_num2 = np.array([8, 12, 16, 24, 32, 48, 64, 80])  #8
        self.filter_num3 = np.array([16, 24, 32, 48, 64, 80, 96, 128])  #8
        self.filter_num4 = np.array(
            [24, 32, 48, 64, 80, 96, 128, 144, 160, 192])  #10
        self.filter_num5 = np.array(
            [32, 48, 64, 80, 96, 128, 144, 160, 192, 224])  #10
        self.filter_num6 = np.array(
            [64, 80, 96, 128, 144, 160, 192, 224, 256, 320, 384, 512])  #12
        # self.k_size means kernel size
        self.k_size = np.array([3, 5])  #2
        # self.multiply means expansion_factor of each _inverted_residual_unit
        self.multiply = np.array([1, 2, 3, 4, 6])  #5
        # self.repeat means repeat_num _inverted_residual_unit in each _invresi_blocks
        self.repeat = np.array([1, 2, 3, 4, 5, 6])  #6
        self.scale = scale

    def init_tokens(self):
        return [0] * (len(self.block_mask) * 4)

    def range_table(self):
        range_table_base = []
        if self.block_mask != None:
            for i in range(len(self.block_mask)):
                filter_num = self.__dict__['filter_num{}'.format(i + 1 if i < 6
                                                                 else 6)]
                range_table_base.append(len(self.multiply))
                range_table_base.append(len(filter_num))
                range_table_base.append(len(self.repeat))
                range_table_base.append(len(self.k_size))
            #[len(self.multiply), len(self.filter_num1), len(self.repeat), len(self.k_size),
            # len(self.multiply), len(self.filter_num1), len(self.repeat), len(self.k_size),
            # len(self.multiply), len(self.filter_num2), len(self.repeat), len(self.k_size),
            # len(self.multiply), len(self.filter_num3), len(self.repeat), len(self.k_size),
            # len(self.multiply), len(self.filter_num4), len(self.repeat), len(self.k_size),
            # len(self.multiply), len(self.filter_num5), len(self.repeat), len(self.k_size),
            # len(self.multiply), len(self.filter_num6), len(self.repeat), len(self.k_size)]
        return range_table_base

    def token2arch(self, tokens=None):
        """
        return mobilenetv2 net_arch function
        """

        if tokens is None:
            tokens = self.init_tokens()
        print(tokens)
        print(len(tokens))

        bottleneck_params_list = []
        if self.block_mask == None:
            if self.block_num >= 1:
                bottleneck_params_list.append(
                    (1, self.head_num[tokens[0]], 1, 1, 3))
            if self.block_num >= 2:
                bottleneck_params_list.append(
                    (self.multiply[tokens[1]], self.filter_num1[tokens[2]],
                     self.repeat[tokens[3]], 2, self.k_size[tokens[4]]))
            if self.block_num >= 3:
                bottleneck_params_list.append(
                    (self.multiply[tokens[5]], self.filter_num1[tokens[6]],
                     self.repeat[tokens[7]], 2, self.k_size[tokens[8]]))
            if self.block_num >= 4:
                bottleneck_params_list.append(
                    (self.multiply[tokens[9]], self.filter_num2[tokens[10]],
                     self.repeat[tokens[11]], 2, self.k_size[tokens[12]]))
            if self.block_num >= 5:
                bottleneck_params_list.append(
                    (self.multiply[tokens[13]], self.filter_num3[tokens[14]],
                     self.repeat[tokens[15]], 2, self.k_size[tokens[16]]))
                bottleneck_params_list.append(
                    (self.multiply[tokens[17]], self.filter_num4[tokens[18]],
                     self.repeat[tokens[19]], 1, self.k_size[tokens[20]]))
            if self.block_num >= 6:
                bottleneck_params_list.append(
                    (self.multiply[tokens[21]], self.filter_num5[tokens[22]],
                     self.repeat[tokens[23]], 2, self.k_size[tokens[24]]))
                bottleneck_params_list.append(
                    (self.multiply[tokens[25]], self.filter_num6[tokens[26]],
                     self.repeat[tokens[27]], 1, self.k_size[tokens[28]]))
        else:
            for i in range(len(self.block_mask)):
                filter_num = self.__dict__['filter_num{}'.format(i + 1 if i < 6
                                                                 else 6)]
                bottleneck_params_list.append(
                    (self.multiply[tokens[i * 4]],
                     filter_num[tokens[i * 4 + 1]],
                     self.repeat[tokens[i * 4 + 2]], 2
                     if self.block_mask[i] == 1 else 1,
                     self.k_size[tokens[i * 4 + 3]]))

        def net_arch(input):
            # all padding is 'SAME' in the conv2d, can compute the actual padding automatic. 
            # bottleneck sequences
            i = 1
            in_c = int(32 * self.scale)
            for layer_setting in bottleneck_params_list:
                t, c, n, s, k = layer_setting
                i += 1
                input = self._invresi_blocks(
                    input=input,
                    in_c=in_c,
                    t=t,
                    c=int(c * self.scale),
                    n=n,
                    s=s,
                    k=k,
                    name='mobilenetv2_conv' + str(i))
                in_c = int(c * self.scale)

            return input

        return net_arch

    def _shortcut(self, input, data_residual):
        """Build shortcut layer.
        Args:
            input(Variable): input.
            data_residual(Variable): residual layer.
        Returns:
            Variable, layer output.
        """
        return fluid.layers.elementwise_add(input, data_residual)

    def _inverted_residual_unit(self,
                                input,
                                num_in_filter,
                                num_filters,
                                ifshortcut,
                                stride,
                                filter_size,
                                expansion_factor,
                                reduction_ratio=4,
                                name=None):
        """Build inverted residual unit.
        Args:
            input(Variable), input.
            num_in_filter(int), number of in filters.
            num_filters(int), number of filters.
            ifshortcut(bool), whether using shortcut.
            stride(int), stride.
            filter_size(int), filter size.
            padding(str|int|list), padding.
            expansion_factor(float), expansion factor.
            name(str), name.
        Returns:
            Variable, layers output.
        """
        num_expfilter = int(round(num_in_filter * expansion_factor))
        channel_expand = conv_bn_layer(
            input=input,
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding='SAME',
            num_groups=1,
            act='relu6',
            name=name + '_expand')

        bottleneck_conv = conv_bn_layer(
            input=channel_expand,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding='SAME',
            num_groups=num_expfilter,
            act='relu6',
            name=name + '_dwise',
            use_cudnn=False)

        linear_out = conv_bn_layer(
            input=bottleneck_conv,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding='SAME',
            num_groups=1,
            act=None,
            name=name + '_linear')
        out = linear_out
        if ifshortcut:
            out = self._shortcut(input=input, data_residual=out)
        return out

    def _invresi_blocks(self, input, in_c, t, c, n, s, k, name=None):
        """Build inverted residual blocks.
        Args:
            input: Variable, input.
            in_c: int, number of in filters.
            t: float, expansion factor.
            c: int, number of filters.
            n: int, number of layers.
            s: int, stride.
            k: int, filter size.
            name: str, name.
        Returns:
            Variable, layers output.
        """
        first_block = self._inverted_residual_unit(
            input=input,
            num_in_filter=in_c,
            num_filters=c,
            ifshortcut=False,
            stride=s,
            filter_size=k,
            expansion_factor=t,
            name=name + '_1')

        last_residual_block = first_block
        last_c = c

        for i in range(1, n):
            last_residual_block = self._inverted_residual_unit(
                input=last_residual_block,
                num_in_filter=last_c,
                num_filters=c,
                ifshortcut=True,
                stride=1,
                filter_size=k,
                expansion_factor=t,
                name=name + '_' + str(i + 1))
        return last_residual_block
