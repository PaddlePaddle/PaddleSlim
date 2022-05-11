#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

from ..core import GraphWrapper
from .patterns_common import *

__all__ = ['preprocess_transformer_patterns']


def _append_transformer_prune_params(op, graph, block_num, params_dict):
    for next_op in graph.next_ops(op):
        if next_op.type() in ['mul', 'matmul', 'matmul_v2'
                              ] and is_dynamic_weight_op(next_op):
            if block_num not in params_dict:
                params_dict[block_num] = {}
                params_dict[block_num]['P1'] = [get_weight(next_op)]
            else:
                params_dict[block_num]['P1'].append(get_weight(next_op))
            params_dict[block_num]['P1'].append(
                get_weight(has_bias(next_op, graph)))
            op = next_op
    next_op = find_weight_op(op, graph)
    if next_op:
        params_dict[block_num]['P2'] = [get_weight(next_op)]
        params_dict[block_num]['P2'].append(
            get_weight(has_bias(next_op, graph)))
    return params_dict


def preprocess_transformer_patterns(patterns, graph):
    """ """
    mha_weight = {}
    ffn_weight = {}
    for pattern_name, pattern_ops in patterns.items():
        if pattern_name == 'input_mask':
            continue
        block_num = int(pattern_name.split('$')[-1])
        if 'MHA' in pattern_name:
            mha_weight = _append_transformer_prune_params(pattern_ops[0], graph,
                                                          block_num, mha_weight)
            mha_weight[block_num]['reshape_op'] = []
            for op in pattern_ops:
                if op.type() in ['reshape', 'reshape2']:
                    mha_weight[block_num]['reshape_op'].append(op)
        elif 'FFN' in pattern_name:
            ffn_weight = _append_transformer_prune_params(pattern_ops[0], graph,
                                                          block_num, ffn_weight)

    return mha_weight, ffn_weight
