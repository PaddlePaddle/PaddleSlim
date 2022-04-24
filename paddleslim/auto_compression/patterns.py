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
###from ..core import GraphWrapper
from paddleslim.core import GraphWrapper
from patterns_common import *


def find_final_nodes(program):
    final_nodes = []
    graph = GraphWrapper(program)
    for op in sorted(graph.ops()):
        if op.type() in ALL_WEIGHT_OP and is_output_weight_ops(op, graph):
            if has_bias(op, graph) != None:
                final_nodes.append(n_op.all_outputs())
            else:
                if op.type() == 'batch_norm':
                    out_var = op.outputs('Y')
                else:
                    out_var = op.all_outpus()
                final_nodes.append(out_var)
    return final_nodes


def get_patterns(program, model_type):
    distill_node = []
    graph = GraphWrapper(program)
    patterns = {}
    block_num = 0
    for op in graph.ops():
        if op.type() == 'elementwise_add':
            inp1, inp2 = op.all_inputs()[0], op.all_inputs()[1]
            if (not inp1._var.persistable) and (not inp2._var.persistable):
                sc_path = []
                shortcut_start_op = []
                is_sc = is_shortcut(op, graph, sc_path, shortcut_start_op)
                if is_sc:
                    shortcut_start_op = shortcut_start_op[0]
                    pattern_ops, pattern_ops_type = bfs(shortcut_start_op,
                                                        graph, op.idx())

                    if model_type == 'transformer' and 'fetch' in pattern_ops_type:
                        if 'input_mask' not in patterns:
                            patterns['input_mask'] = pattern_ops[0]._op

                    if 'fetch' in pattern_ops_type:
                        continue

                    out_var_name = op.all_outputs()[0]._var.name
                    if model_type == 'transformer':
                        if 'softmax' in pattern_ops_type:
                            patterns['MHA$' + str(block_num)] = pattern_ops
                        else:  ##### is FFN
                            patterns['FFN$' + str(block_num)] = pattern_ops
                            block_num += 1
                            distill_node.append('teacher_' + out_var_name)
                            distill_node.append(out_var_name)
                    else:
                        patterns[shortcut_start_op.type() + '$' + str(op.idx(
                        ))] = pattern_ops
                        distill_node.append('teacher_' + out_var_name)
                        distill_node.append(out_var_name)

    return patterns, graph, distill_node
