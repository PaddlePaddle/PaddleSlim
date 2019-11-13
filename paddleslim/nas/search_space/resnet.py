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
                 scale=1.0,
                 class_dim=1000):
        super(ResNetSpace, self).__init__(input_size, output_size, block_num)
        pass

    def init_tokens(self):
        return [0, 0, 0, 0, 0, 0]

    def range_table(self):
        return [2, 2, 2, 2, 2, 2]

    def token2arch(self, tokens=None):
        if tokens is None:
            self.init_tokens()

        def net_arch(input):
            input = conv_bn_layer(
                input,
                num_filters=32,
                filter_size=3,
                stride=2,
                padding='SAME',
                act='sigmoid',
                name='resnet_conv1_1')

            return input

        return net_arch
