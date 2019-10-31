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

from SearchSpace import SearchSpace
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid as fluid
import numpy as np

class MobileNetV2Space(SearchSpace):
    def __init__(self, input_size, output_size, block_num, scale=1.0, class_dim=1000):
        super(MobileNetV2Space, self).__init__(input_size, output_size, block_num)
        self.scale=scale
        self.head_num = np.array([3,4,8,12,16,24,32]) #7
        self.filter_num1 = np.array([3,4,8,12,16,24,32,48]) #8
        self.filter_num2 = np.array([8,12,16,24,32,48,64,80]) #8
        self.filter_num3 = np.array([16,24,32,48,64,80,96,128]) #8
        self.filter_num4 = np.array([24,32,48,64,80,96,128,144,160,192]) #10
        self.filter_num5 = np.array([32,48,64,80,96,128,144,160,192,224]) #10
        self.filter_num6 = np.array([64,80,96,128,144,160,192,224,256,320,384,512]) #12
        self.k_size = np.array([3,5]) #2
        self.multiply = np.array([1,2,3,4,6]) #5
        self.repeat = np.array([1,2,3,4,5,6]) #6
        self.class_dim=class_dim

    def init_tokens(self):
        """
        initial tokens
        """
        # original MobileNetV2
        return [4,          # 1, 16, 1
                4, 5, 1, 0, # 6, 24, 1
                4, 5, 1, 0, # 6, 24, 2
                4, 4, 2, 0, # 6, 32, 3
                4, 4, 3, 0, # 6, 64, 4
                4, 5, 2, 0, # 6, 96, 3
                4, 7, 2, 0, # 6, 160, 3
                4, 9, 0, 0] # 6, 320, 1

    def range_table(self):
        """
        get range table of current search space 
        """
        # head_num + 7 * [multiple(expansion_factor), filter_num, repeat, kernel_size]
        return [7, 
                5, 8, 6, 2,
                5, 8, 6, 2,
                5, 8, 6, 2,
                5, 8, 6, 2,
                5, 10, 6, 2,
                5, 10, 6, 2,
                5, 12, 6, 2]

    def token2arch(self, tokens=None):
        """
        return netArch function
        """
        if tokens is None:
            tokens = self.init_tokens()

        bottleneck_params_list = [
            (1, self.head_num[tokens[0]], 1, 1, 3),
            (self.multiply[tokens[1]], self.filter_num1[tokens[2]], self.repeat[tokens[3]], 2, self.k_size[tokens[4]]),
            (self.multiply[tokens[5]], self.filter_num1[tokens[6]], self.repeat[tokens[7]], 2, self.k_size[tokens[8]]),
            (self.multiply[tokens[9]], self.filter_num2[tokens[10]], self.repeat[tokens[11]], 2, self.k_size[tokens[12]]),
            (self.multiply[tokens[13]], self.filter_num3[tokens[14]], self.repeat[tokens[15]], 2, self.k_size[tokens[16]]),
            (self.multiply[tokens[17]], self.filter_num3[tokens[18]], self.repeat[tokens[19]], 1, self.k_size[tokens[20]]),
            (self.multiply[tokens[21]], self.filter_num5[tokens[22]], self.repeat[tokens[23]], 2, self.k_size[tokens[24]]),
            (self.multiply[tokens[25]], self.filter_num6[tokens[26]], self.repeat[tokens[27]], 1, self.k_size[tokens[28]]),
        ]
        bottleneck_params_list = bottleneck_params_list[:self.block_num]

        def netArch(input):
            #conv1
            input = self.conv_bn_layer(
                input,
                num_filters=int(32 * self.scale),
                filter_size=3,
                stride=2,
                padding=1,
                if_act=True,
                name='conv1_1')

            # bottleneck sequences
            i = 1
            in_c = int(32 * self.scale)
            for layer_setting in bottleneck_params_list:
                t, c, n, s, k = layer_setting
                i += 1
                input = self.invresi_blocks(
                    input=input,
                    in_c=in_c,
                    t=t,
                    c=int(c * self.scale),
                    n=n,
                    s=s,
                    k=k,
                    name='conv' + str(i))
                in_c = int(c * self.scale)
            #last_conv
            input = self.conv_bn_layer(
                input=input,
                num_filters=int(1280 * self.scale) if self.scale > 1.0 else 1280,
                filter_size=1,
                stride=1,
                padding=0,
                if_act=True,
                name='conv9')

            #input = fluid.layers.pool2d(
            #    input=input,
            #    pool_size=7,
            #    pool_stride=1,
            #    pool_type='avg',
            #    global_pooling=True)

            #output = fluid.layers.fc(input=input,
            #                         size=self.class_dim,
            #                         param_attr=ParamAttr(name='fc10_weights'),
            #                         bias_attr=ParamAttr(name='fc10_offset'))
            return input

        return netArch


    def conv_bn_layer(self, input, filter_size, num_filters, stride, padding, num_groups=1, if_act=True, name=None, use_cudnn=True):
        """Build convolution and batch normalization layers.
        Args:
            input: Variable, input.
            filter_size: int, filter size.
            num_filters: int, number of filters.
            stride: int, stride.
            padding: int, padding.
            num_groups: int, number of groups.
            if_act: bool, whether using activation.
            name: str, name.
            use_cudnn: bool, whether use cudnn.
        Returns:
            Variable, layers output.
        """
        conv = fluid.layers.conv2d(input, num_filters=num_filters, filter_size=filter_size, stride=stride, padding=padding, 
                                   groups=num_groups, act=None, use_cudnn=use_cudnn, param_attr=ParamAttr(name=name+'_weights'), bias_attr=False)
        bn_name = name + '_bn'
        bn = fluid.layers.batch_norm(input=conv, param_attr=ParamAttr(name=bn_name+'_scale'), bias_attr=ParamAttr(name=bn_name+'_offset'),
                                     moving_mean_name=bn_name+'_mean', moving_variance_name=bn_name+'_variance')
        if if_act:
            return fluid.layers.relu6(bn)
        else:
            return bn

    def shortcut(self, input, data_residual):
        """Build shortcut layer.
        Args:
            input: Variable, input.
            data_residual: Variable, residual layer.
        Returns:
            Variable, layer output.
        """
        return fluid.layers.elementwise_add(input, data_residual)


    def inverted_residual_unit(self,
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
            input: Variable, input.
            num_in_filter: int, number of in filters.
            num_filters: int, number of filters.
            ifshortcut: bool, whether using shortcut.
            stride: int, stride.
            filter_size: int, filter size.
            padding: int, padding.
            expansion_factor: float, expansion factor.
            name: str, name.
        Returns:
            Variable, layers output.
        """
        num_expfilter = int(round(num_in_filter * expansion_factor))
        channel_expand = self.conv_bn_layer(
            input=input,
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            name=name + '_expand')

        bottleneck_conv = self.conv_bn_layer(
            input=channel_expand,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) / 2),
            num_groups=num_expfilter,
            if_act=True,
            name=name + '_dwise',
            use_cudnn=False)

        linear_out = self.conv_bn_layer(
            input=bottleneck_conv,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=False,
            name=name + '_linear')
        out = linear_out
        if ifshortcut:
            out = self.shortcut(input=input, data_residual=out)
        return out

    def invresi_blocks(self,
                       input,
                       in_c,
                       t,
                       c,
                       n,
                       s,
                       k,
                       name=None):
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
        first_block = self.inverted_residual_unit(
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
            last_residual_block = self.inverted_residual_unit(
                input=last_residual_block,
                num_in_filter=last_c,
                num_filters=c,
                ifshortcut=True,
                stride=1,
                filter_size=k,
                expansion_factor=t,
                name=name + '_' + str(i + 1))
        return last_residual_block


