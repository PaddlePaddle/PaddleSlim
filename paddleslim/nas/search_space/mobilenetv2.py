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

__all__ = ["MobileNetV2Space"]


@SEARCHSPACE.register
class MobileNetV2Space(SearchSpaceBase):
    def __init__(self,
                 input_size,
                 output_size,
                 block_num,
                 block_mask=None,
                 scale=1.0,
                 class_dim=1000):
        super(MobileNetV2Space, self).__init__(input_size, output_size,
                                               block_num, block_mask)
        assert self.block_mask == None, 'MobileNetV2Space will use origin MobileNetV2 as seach space, so use input_size, output_size and block_num to search'
        # self.head_num means the first convolution channel
        self.head_num = np.array([3, 4, 8, 12, 16, 24, 32])  #7
        # self.filter_num1 ~ self.filter_num6 means following convlution channel
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
        self.class_dim = class_dim

        assert self.block_num < 7, 'MobileNetV2: block number must less than 7, but receive block number is {}'.format(
            self.block_num)

    def init_tokens(self):
        """
        The initial token.
        The first one is the index of the first layers' channel in self.head_num,
        each line in the following represent the index of the [expansion_factor, filter_num, repeat_num, kernel_size]
        """
        # original MobileNetV2
        # yapf: disable
        init_token_base =  [4,          # 1, 16, 1
                4, 5, 1, 0, # 6, 24, 1
                4, 5, 1, 0, # 6, 24, 2
                4, 4, 2, 0, # 6, 32, 3
                4, 4, 3, 0, # 6, 64, 4
                4, 5, 2, 0, # 6, 96, 3
                4, 7, 2, 0, # 6, 160, 3
                4, 9, 0, 0] # 6, 320, 1
        # yapf: enable

        if self.block_num < 5:
            self.token_len = 1 + (self.block_num - 1) * 4
        else:
            self.token_len = 1 + (self.block_num + 2 *
                                  (self.block_num - 5)) * 4

        return init_token_base[:self.token_len]

    def range_table(self):
        """
        Get range table of current search space, constrains the range of tokens. 
        """
        # head_num + 7 * [multiple(expansion_factor), filter_num, repeat, kernel_size]
        # yapf: disable
        range_table_base =  [len(self.head_num),
                len(self.multiply), len(self.filter_num1), len(self.repeat), len(self.k_size),
                len(self.multiply), len(self.filter_num1), len(self.repeat), len(self.k_size),
                len(self.multiply), len(self.filter_num2), len(self.repeat), len(self.k_size),
                len(self.multiply), len(self.filter_num3), len(self.repeat), len(self.k_size),
                len(self.multiply), len(self.filter_num4), len(self.repeat), len(self.k_size),
                len(self.multiply), len(self.filter_num5), len(self.repeat), len(self.k_size),
                len(self.multiply), len(self.filter_num6), len(self.repeat), len(self.k_size)]
        range_table_base = list(np.array(range_table_base) - 1)
        # yapf: enable
        return range_table_base[:self.token_len]

    def token2arch(self, tokens=None):
        """
        return net_arch function
        """

        if tokens is None:
            tokens = self.init_tokens()

        bottleneck_params_list = []
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

        def net_arch(input, end_points=None, decode_points=None):
            decode_ends = dict()

            def check_points(count, points):
                if points is None:
                    return False
                else:
                    if isinstance(points, list):
                        return (True if count in points else False)
                    else:
                        return (True if count == points else False)

            #conv1
            # all padding is 'SAME' in the conv2d, can compute the actual padding automatic. 
            input = conv_bn_layer(
                input,
                num_filters=int(32 * self.scale),
                filter_size=3,
                stride=2,
                padding='SAME',
                act='relu6',
                name='mobilenetv2_conv1_1')
            layer_count = 1
            if check_points(layer_count, decode_points):
                decode_ends[layer_count] = input

            if check_points(layer_count, end_points):
                return input, decode_ends

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
                layer_count += n

                if check_points(layer_count, decode_points):
                    decode_ends[layer_count] = depthwise_output

                if check_points(layer_count, end_points):
                    return input, decode_ends

            # last conv
            input = conv_bn_layer(
                input=input,
                num_filters=int(1280 * self.scale)
                if self.scale > 1.0 else 1280,
                filter_size=1,
                stride=1,
                padding='SAME',
                act='relu6',
                name='mobilenetv2_conv' + str(i + 1))

            input = fluid.layers.pool2d(
                input=input,
                pool_size=7,
                pool_stride=1,
                pool_type='avg',
                global_pooling=True,
                name='mobilenetv2_last_pool')

            # if output_size is 1, add fc layer in the end
            if self.output_size == 1:
                input = fluid.layers.fc(
                    input=input,
                    size=self.class_dim,
                    param_attr=ParamAttr(name='mobilenetv2_fc_weights'),
                    bias_attr=ParamAttr(name='mobilenetv2_fc_offset'))
            else:
                assert self.output_size == input.shape[2], \
                          ("output_size must EQUAL to input_size / (2^block_num)."
                          "But receive input_size={}, output_size={}, block_num={}".format(
                          self.input_size, self.output_size, self.block_num))

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

        depthwise_output = bottleneck_conv

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
        return out, depthwise_output

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
            last_residual_block, depthwise_output = self._inverted_residual_unit(
                input=last_residual_block,
                num_in_filter=last_c,
                num_filters=c,
                ifshortcut=True,
                stride=1,
                filter_size=k,
                expansion_factor=t,
                name=name + '_' + str(i + 1))
        return last_residual_block, depthwise_output
