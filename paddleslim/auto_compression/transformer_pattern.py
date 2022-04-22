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

from paddleslim.core import GraphWrapper

ALL_WEIGHT_OP = [
    'conv2d', 'mul', 'matmul', 'embedding', 'conv2d_transpose',
    'depthwise_conv2d', 'batch_norm', 'layer_norm', 'instance_norm',
    'sync_batch_norm', 'matmul_v2'
]


def _find_next_target_op(op, graph, target_op_idx, sc_path):
    if op.idx() == target_op_idx:
        return True
    n_ops = graph.next_ops(op)
    for n_op in n_ops:
        sc_path.append(n_op.type())
        return _find_next_target_op(n_op, graph, target_op_idx, sc_path)
    return False


def is_shortcut(op, graph, sc_path, shortcut_start_op):
    """
       op /```````````````````\ add
          \____op1___op2__..._/
    """
    inps = op.all_inputs()
    pre_ops = graph.pre_ops(op)
    for p_op in pre_ops:
        n_ops = graph.next_ops(p_op)
        if len(n_ops) == 1:
            continue
        ### note: only support one branch donnot have op
        has_sc = False
        for n_op in n_ops:
            if n_op.idx() == op.idx():
                shortcut_start_op.append(p_op)
                has_sc = True
        if has_sc:
            for n_op in n_ops:
                if n_op.idx() != op.idx():
                    sc_path.append(p_op.type())
                    sc_path.append(n_op.type())
                    return _find_next_target_op(n_op, graph, op.idx(), sc_path)
    return False


def bfs(op, graph, target_op_idx):
    pattern_ops = []
    pattern_ops_type = []
    visited = []
    pq = [op]
    while pq:
        cnt = len(pq)
        level = []
        for _ in range(cnt):
            cur = pq.pop(0)
            level.append(cur.type())
            if cur.idx() not in visited:
                ### first op must be start op
                pattern_ops.append(cur)
                pattern_ops_type.append(cur.type())
                visited.append(cur.idx())
            for n_op in graph.next_ops(cur):
                if n_op.idx() == target_op_idx or n_op.idx() in visited:
                    continue
                pq.append(n_op)
    return pattern_ops, pattern_ops_type


def _find_weight_op(op, graph):
    """ Find operators with weight.
    """
    next_ops = sorted(graph.next_ops(op))
    for next_op in next_ops:
        if _is_dynamic_weight_op(next_op):
            return next_op
        else:
            return _find_weight_op(next_op, graph)


def _get_weight(op, return_name=True):
    for inp in op.all_inputs():
        if inp._var.persistable == True:
            if return_name:
                return inp.name()
            else:
                return inp


def _is_dynamic_weight_op(op):
    weight_ops = ALL_WEIGHT_OP
    if op.type() in weight_ops:
        if op.type() in ['mul', 'matmul', 'matmul_v2']:
            for inp in sorted(op.all_inputs()):
                if inp._var.persistable == True:
                    return True
            return False
        return True
    return False


def _is_output_weight_ops(op, graph):
    next_ops = sorted(graph.next_ops(op))
    for next_op in next_ops:
        if _is_dynamic_weight_op(next_op):
            return False
        return _is_output_weight_ops(next_op, graph)
    return True


def _has_bias(op, graph):
    n_op = graph.next_ops(op)[0]
    if op.type() in ALL_WEIGHT_OP:
        if n_op.type() == 'elementwise_add':
            for inp in n_op.all_inputs():
                if inp._var.persistable == True:
                    return n_op
    return None


def _append_transformer_prune_params(op, graph, block_num, params_dict):
    for next_op in graph.next_ops(op):
        if next_op.type() in ['mul', 'matmul', 'matmul_v2'
                              ] and _is_dynamic_weight_op(next_op):
            if block_num not in params_dict:
                params_dict[block_num] = {}
                params_dict[block_num]['P1'] = [_get_weight(next_op)]
            else:
                params_dict[block_num]['P1'].append(_get_weight(next_op))
            params_dict[block_num]['P1'].append(
                _get_weight(_has_bias(next_op, graph)))
            op = next_op
    next_op = _find_weight_op(op, graph)
    if next_op:
        params_dict[block_num]['P2'] = [_get_weight(next_op)]
        params_dict[block_num]['P2'].append(
            _get_weight(_has_bias(next_op, graph)))
    return params_dict


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
                    out_var_name = op.all_outputs()[0]._var.name
                    ### distill_node.append('teacher_'+out_var_name)
                    ### distill_node.append(out_var_name)
                    pattern_ops, pattern_ops_type = bfs(shortcut_start_op,
                                                        graph, op.idx())

                    if model_type == 'transformer' and 'stack' in pattern_ops_type:
                        for op in pattern_ops:
                            if op.type() == 'stack':
                                if 'stack' not in patterns:
                                    patterns['stack'] = op._op

                    if 'fetch' in pattern_ops_type:
                        continue

                    if model_type == 'transformer':
                        if 'softmax' in pattern_ops_type:
                            patterns['MHA$' + str(block_num)] = pattern_ops
                        else:  ##### is FFN
                            patterns['FFN$' + str(block_num)] = pattern_ops
                            block_num += 1
                    else:
                        patterns[shortcut_start_op.type() + '$' + str(op.idx(
                        ))] = pattern_ops

    return patterns, graph


def preprocess_transformer_patterns(patterns, graph):
    mha_weight = {}
    ffn_weight = {}
    for pattern_name, pattern_ops in patterns.items():
        if pattern_name == 'stack':
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
