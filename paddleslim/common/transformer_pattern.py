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


def _find_gemm_op(op, graph):
    while op.type() not in ['mul', 'matmul', 'matmul_v2']:
        next_op = find_weight_op(op, graph)
        op = next_op
    return op


def _append_transformer_prune_params(op_lists, graph, block_num, params_dict):
    first_op = op_lists[0]
    for next_op in graph.next_ops(first_op):
        if next_op.type() == 'elementwise_add':
            continue
        next_op = _find_gemm_op(next_op, graph)
        if next_op.type() in [
                'mul', 'matmul', 'matmul_v2'
        ] and has_trainable_var(next_op) and next_op in op_lists:
            if block_num not in params_dict:
                params_dict[block_num] = {}
                params_dict[block_num]['P1'] = [get_weight(next_op)]
            else:
                params_dict[block_num]['P1'].append(get_weight(next_op))
            params_dict[block_num]['P1'].append(
                get_weight(has_bias(next_op, graph)))
            op = next_op
    next_op = _find_gemm_op(find_weight_op(op, graph), graph)
    if next_op and next_op in op_lists:
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
            mha_weight = _append_transformer_prune_params(
                pattern_ops, graph, block_num, mha_weight)
            mha_weight[block_num]['reshape_op'] = []
            for op in pattern_ops:
                if op.type() in ['reshape', 'reshape2']:
                    mha_weight[block_num]['reshape_op'].append(op)
        elif 'FFN' in pattern_name:
            ffn_weight = _append_transformer_prune_params(
                pattern_ops, graph, block_num, ffn_weight)

    return mha_weight, ffn_weight
