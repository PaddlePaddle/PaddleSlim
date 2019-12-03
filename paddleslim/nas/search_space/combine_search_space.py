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
from .search_space_registry import SEARCHSPACE
from .base_layer import conv_bn_layer

__all__ = ["CombineSearchSpace"]


class CombineSearchSpace(object):
    """
    Combine Search Space.
    Args:
        configs(list<tuple>): multi config.
    """

    def __init__(self, config_lists):
        self.lens = len(config_lists)
        self.spaces = []
        for config_list in config_lists:
            key, config = config_list
            self.spaces.append(self._get_single_search_space(key, config))
        self.init_tokens()

    def _get_single_search_space(self, key, config):
        """
        get specific model space based on key and config.

        Args:
            key(str): model space name.
            config(dict): basic config information.
        return:
            model space(class)
        """
        cls = SEARCHSPACE.get(key)
        block_mask = config['block_mask'] if 'block_mask' in config else None
        space = cls(config['input_size'],
                    config['output_size'],
                    config['block_num'],
                    block_mask=block_mask)
        return space

    def init_tokens(self):
        """
        Combine init tokens.
        """
        tokens = []
        self.single_token_num = []
        for space in self.spaces:
            tokens.extend(space.init_tokens())
            self.single_token_num.append(len(space.init_tokens()))
        return tokens

    def range_table(self):
        """
        Combine range table.
        """
        range_tables = []
        for space in self.spaces:
            range_tables.extend(space.range_table())
        return range_tables

    def token2arch(self, tokens=None):
        """
        Combine model arch
        """
        if tokens is None:
            tokens = self.init_tokens()

        token_list = []
        start_idx = 0
        end_idx = 0

        for i in range(len(self.single_token_num)):
            end_idx += self.single_token_num[i]
            token_list.append(tokens[start_idx:end_idx])
            start_idx = end_idx

        model_archs = []
        for space, token in zip(self.spaces, token_list):
            model_archs.append(space.token2arch(token))

        return model_archs
