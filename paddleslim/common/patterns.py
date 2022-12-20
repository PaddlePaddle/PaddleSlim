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

import logging
import os
import sys
import numpy as np
import warnings

import paddle
from ..core import GraphWrapper
from .patterns_common import *

__all__ = ['find_final_nodes', 'get_patterns']


def find_final_nodes(program):
    """ Find the output of the final op with weights in the program """
    final_nodes = []
    graph = GraphWrapper(program)
    for op in sorted(graph.ops()):
        if has_trainable_var(op) and is_final_op_with_trainable_var(op, graph):
            n_op = has_bias(op, graph)
            if n_op is not None:
                final_nodes.extend(n_op.all_outputs())
            else:
                if op.type() == 'batch_norm':
                    out_var = op.outputs('Y')
                else:
                    out_var = op.all_outputs()
                final_nodes.extend(out_var)
    return final_nodes


def _is_mha(pattern_ops, pattern_ops_type, skip_quant_tensor_list=[]):
    """ judge whether this pattern is multihead attention """
    if pattern_ops_type.count('softmax') != 1 or pattern_ops_type.count(
            'fetch') > 0:
        return False

    matmul_num = 0
    for op in pattern_ops:
        if op.type() in ['matmul', 'matmul_v2']:
            if not has_trainable_var(op):
                matmul_num += 1
    if matmul_num == 2:
        return True
    return False


def _is_ffn(pattern_ops, pattern_ops_type):
    """ judge whether this pattern is feed forward network """
    if pattern_ops_type.count('layer_norm') != 1:
        return False

    linear_num = 0
    act_num = 0
    for op in pattern_ops:
        if op.type() in ['mul', 'matmul', 'matmul_v2']:
            if has_trainable_var(op):
                linear_num += 1
        if op.type() in ['relu', 'gelu']:
            act_num += 1
    if linear_num == 2 and act_num == 1:
        return True

    return False


def get_patterns(program, only_final_node=True):
    """ distinguish the pattern in the program and get model type """
    skip_quant_tensor_list = []
    patterns = {}
    graph = GraphWrapper(program)
    block_num = 0
    model_type = None
    for op in graph.ops():
        if len(op.all_inputs()) == 0 or op.all_inputs()[0] is None:
            continue
        belonged_teacher = False
        for inp in op.all_inputs():
            if 'teacher' in inp._var.name:
                belonged_teacher = True
                break
        if belonged_teacher:
            continue

        if op.type() == 'elementwise_add':
            inp1, inp2 = op.all_inputs()[0], op.all_inputs()[1]
            if (not inp1._var.persistable) and (not inp2._var.persistable):
                sc_path = []
                shortcut_start_op = []
                is_sc, target_op_idx = is_shortcut(op, graph, sc_path,
                                                   shortcut_start_op)
                if is_sc:
                    out_var_name = op.all_outputs()[0]._var.name

                    shortcut_start_op = shortcut_start_op[0]
                    next_ops = graph.next_ops(op)
                    pattern_ops, pattern_ops_type = traversal_ops(
                        shortcut_start_op, graph, target_op_idx)

                    pattern_name = shortcut_start_op.type() + '$' + str(op.idx(
                    ))

                    if _is_mha(pattern_ops, pattern_ops_type,
                               skip_quant_tensor_list):
                        model_type = 'transformer'
                        pattern_name = 'MHA$' + str(block_num)

                    if model_type == 'transformer' and _is_ffn(
                            pattern_ops, pattern_ops_type):
                        pattern_name = 'FFN$' + str(block_num)
                        block_num += 1

                    if model_type == 'transformer' and (
                            'fetch' in pattern_ops_type or
                            pattern_ops_type[-1] == 'scale'):
                        if 'input_mask' not in patterns:
                            patterns['input_mask'] = pattern_ops[0]._op

                    if 'fetch' in pattern_ops_type or pattern_ops_type[
                            -1] == 'scale':
                        continue

                    patterns[pattern_name] = pattern_ops

    #### skip quant matmul in attention
    if model_type == 'transformer':
        for block_id in range(len(program.blocks)):
            for op in program.blocks[block_id].ops:
                for inp_name in op.input_arg_names:
                    if inp_name in skip_quant_tensor_list:
                        op._set_attr("op_namescope", "skip_quant")

    return patterns, model_type
