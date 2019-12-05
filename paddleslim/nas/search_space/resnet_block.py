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
from .utils import compute_downsample_num

__all__ = ["ResNetBlockSpace"]


@SEARCHSPACE.register
class ResNetBlockSpace(SearchSpaceBase):
    def __init__(input_size, output_size, block_num, block_mask):
        super(ResNetSpace, self).__init__(input_size, output_size, block_num,
                                          block_mask)
        # use input_size and output_size to compute self.downsample_num
        self.downsample_num = compute_downsample_num(self.input_size,
                                                     self.output_size)
        if self.block_num != None:
            assert self.downsample_num <= self.block_num, 'downsample numeber must be LESS THAN OR EQUAL TO block_num, but NOW: downsample numeber is {}, block_num is {}'.format(
                self.downsample_num, self.block_num)
        self.filter_num = np.array(
            [48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512, 640])
        ### TODO: use repeat to compute normal cell
        #self.repeat = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24]
        self.k_size = np.array([3, 5])

    def init_tokens(self):
        if self.block_mask != None:
            return [0] * (len(self.block_mask) * 2)
        else:
            return [0] * (self.block_num * 2)

    def range_table(self):
        range_table_base = []

        return range_table_base

    def token2arch(self, tokens=None):
        if tokens == None:
            tokens = self.init_tokens()

        self.bottleneck_params_list = []
        if self.block_mask != None:
            for i in range(len(self.block_mask)):
                self.bottleneck_params_list.append(
                    (self.num_filters[tokens[i * 2]],
                     self.kernel_size[tokens[i * 2 + 1]], 2
                     if self.block_mask[i] == 1 else 1))
        else:
            repeat_num = self.block_num / self.downsample_num
            num_minus = self.block_num % self.downsample_num
            for i in range(self.downsample_num):
                self.bottleneck_params_list.append(
                    self.num_filters[tokens[i * 2]],
                    self.kernel_size[tokens[i * 2 + 1]], 2)
                for k in range(repeat_num - 1):
                    kk = k * self.downsample_num + i
                    self.bottleneck_params_list.append(
                        self.num_filters[tokens[kk * 2]],
                        self.kernel_size[tokens[kk * 2 + 1]], 1)
                if self.downsample_num - i <= num_minus:
                    j = self.downsample_num * repeat_num + i
                    self.bottleneck_params_list.append(
                        self.num_filters[tokens[j * 2]],
                        self.kernel_size[tokens[j * 2 + 1]], 1)

            if self.downsample_num == 0 and self.block_num != 0:
                for i in range(len(self.block_num)):
                    self.bottleneck_params_list.append(
                        self.num_filters[tokens[i * 2]],
                        self.kernel_size[tokens[i * 2 + 1]], 1)

        def net_arch(input, return_mid_layer=False, return_block=[]):
            assert isinstance(return_block,
                              list), 'return_block must be a list.'
            layer_count = 0
            mid_layer = dict()
            for layer_setting in self.bottleneck_params_list:
                filter_num, k_size, stride = layer_setting
                if stride == 2:
                    layer_count += 1
                if (layer_count - 1) in return_block:
                    mid_layer[layer_count - 1] = input

                input = self._bottleneck_block(
                    input=input,
                    num_filters=filter_num,
                    kernel_size=k_size,
                    stride=stride,
                    name='resnet' + str(i + 1))

            if return_mid_layer:
                return input, mid_layer
            else:
                return input,

        return net_arch

    def _shortcut(self, input, ch_out, stride, name=None):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return conv_bn_layer(
                input=input,
                filter_size=1,
                num_filters=ch_out,
                stride=stride,
                name=name + '_shortcut')
        else:
            return input

    def _bottleneck_block(self,
                          input,
                          num_filters,
                          kernel_size,
                          stride,
                          name=None):
        conv0 = conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + '_bottleneck_conv0')
        conv1 = conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=kernel_size,
            stride=stride,
            act='relu',
            name=name + '_bottleneck_conv1')
        conv2 = conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + '_bottleneck_conv2')

        short = self._shortcut(input, num_filters * 4, stride, name=name)

        return fluid.layers.elementwise_add(
            x=short, y=conv2, act='relu', name=name + '_bottleneck_add')
