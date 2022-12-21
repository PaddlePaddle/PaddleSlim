import os
import sys
sys.setrecursionlimit(10000)

ALL_WEIGHT_OP = [
    'conv2d', 'mul', 'matmul', 'embedding', 'conv2d_transpose',
    'depthwise_conv2d', 'batch_norm', 'layer_norm', 'instance_norm',
    'sync_batch_norm', 'matmul_v2'
]


def traversal_ops(op, graph, target_op_idx):
    """ Get all operators in the multi-path from op to target op. """
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
                if n_op.is_opt_op() or n_op.is_bwd_op():
                    break
                if n_op.idx() == target_op_idx or n_op.idx() in visited:
                    continue
                pq.append(n_op)
    return pattern_ops, pattern_ops_type


def find_weight_op(op, graph):
    """ Find operators with weight."""
    next_ops = sorted(graph.next_ops(op))
    for next_op in next_ops:
        if has_trainable_var(next_op):
            return next_op
        else:
            return find_weight_op(next_op, graph)


def get_weight(op, return_name=True):
    """ get the weight of operators with weight."""
    for inp in op.all_inputs():
        if inp._var.persistable == True:
            if return_name:
                return inp.name()
            else:
                return inp
    return None


def has_trainable_var(op):
    """ Judge whether the operator with trainable variable """
    weight_ops = ALL_WEIGHT_OP
    if op.type() in weight_ops:
        for inp in sorted(op.all_inputs()):
            if inp._var.persistable == True:
                return True
        return False
    return False


def is_final_op_with_trainable_var(op, graph):
    """ Judge whether is the final op with weights in the graph """
    next_ops = sorted(graph.next_ops(op))
    for next_op in next_ops:
        if has_trainable_var(next_op):
            return False
        return is_final_op_with_trainable_var(next_op, graph)
    return True


def has_bias(op, graph):
    """ Get the bias of the op if exists  """
    n_op = graph.next_ops(op)[0]
    if op.type() in ALL_WEIGHT_OP:
        if n_op.type() == 'elementwise_add':
            for inp in n_op.all_inputs():
                if inp._var.persistable == True:
                    return n_op
    return None


def _find_next_target_op(op, graph, target_op_idx, sc_path):
    """ Find the target op from other branch in the shortcut """
    if op.idx() == target_op_idx:
        return True
    n_ops = graph.next_ops(op)
    for n_op in n_ops:
        sc_path.append(n_op.type())
        return _find_next_target_op(n_op, graph, target_op_idx, sc_path)
    return False


def _is_identity_op(op):
    if op.type() == 'scale' and op.attr('scale') == 1:
        return True
    return False


def is_shortcut(op, graph, sc_path, shortcut_start_op):
    """
       op /```````````````````\\ add
          \\____op1___op2__..._/
    """
    inps = op.all_inputs()
    pre_ops = graph.pre_ops(op)
    for p_op in pre_ops:
        if _is_identity_op(p_op):
            p_op = graph.pre_ops(p_op)[0]
        n_ops = graph.next_ops(p_op)
        if len(n_ops) == 1:
            continue
        ### note: only support one branch donnot have op or has one scale op
        has_sc = False
        for n_op in n_ops:
            if _is_identity_op(n_op):
                n_op = graph.next_ops(n_op)[0]
            if n_op.idx() == op.idx():
                shortcut_start_op.append(p_op)
                has_sc = True
        if has_sc:
            for n_op in n_ops:
                if n_op.idx() != op.idx():
                    sc_path.append(p_op.type())
                    sc_path.append(n_op.type())
                    return _find_next_target_op(n_op, graph, op.idx(),
                                                sc_path), op.idx()
    return False, -1
