#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import paddle
from .layers_base import BaseBlock

__all__ = ['check_search_space']

DYNAMIC_WEIGHT_OP = [
    'conv2d', 'mul', 'matmul', 'embedding', 'conv2d_transpose',
    'depthwise_conv2d', 'matmul_v2'
]

CONV_TYPES = [
    'conv2d', 'conv3d', 'conv1d', 'superconv2d', 'supergroupconv2d',
    'superdepthwiseconv2d', 'matmul_v2'
]

ALL_WEIGHT_OP = [
    'conv2d', 'mul', 'matmul', 'elementwise_add', 'embedding',
    'conv2d_transpose', 'depthwise_conv2d', 'batch_norm', 'layer_norm',
    'instance_norm', 'sync_batch_norm', 'matmul_v2'
]


def _is_dynamic_weight_op(op, all_weight_op=False):
    if all_weight_op == True:
        weight_ops = ALL_WEIGHT_OP
    else:
        weight_ops = DYNAMIC_WEIGHT_OP
    if op.type() in weight_ops:
        if op.type() in ['mul', 'matmul']:
            for inp in sorted(op.all_inputs()):
                if inp._var.persistable == True:
                    return True
            return False
        return True
    return False


def get_actual_shape(transform, channel):
    if transform == None:
        channel = int(channel)
    else:
        if isinstance(transform, float):
            channel = int(channel * transform)
        else:
            channel = int(transform)
    return channel


def _is_depthwise(op):
    """Check if this op is depthwise conv. Only Cin == Cout == groups be consider as depthwise conv.
       The shape of input and the shape of output in depthwise conv must be same in superlayer,
       so depthwise op cannot be consider as weight op
    """
    #if op.type() == 'depthwise_conv2d': ### depthwise_conv2d in paddle is Cout % Cin =0
    #    return True
    if 'conv' in op.type():
        for inp in op.all_inputs():
            if inp._var.persistable and (
                    op.attr('groups') == inp._var.shape[0] and
                    op.attr('groups') * inp._var.shape[1] == inp._var.shape[0]):
                return True
    return False


def _find_weight_ops(op, graph, weights):
    """ Find the vars come from operators with weight.
    """
    pre_ops = sorted(graph.pre_ops(op))
    for pre_op in pre_ops:
        ### if depthwise conv is one of elementwise's input,
        ### add it into this same search space
        if _is_depthwise(pre_op):
            for inp in pre_op.all_inputs():
                if inp._var.persistable:
                    weights.append(inp._var.name)

        if _is_dynamic_weight_op(pre_op) and not _is_depthwise(pre_op):
            for inp in pre_op.all_inputs():
                if inp._var.persistable:
                    weights.append(inp._var.name)
            return weights
        return _find_weight_ops(pre_op, graph, weights)
    return weights


def _find_pre_elementwise_op(op, graph):
    """ Find precedors of the elementwise_add operator in the model.
    """
    same_prune_before_elementwise_add = []
    pre_ops = sorted(graph.pre_ops(op))
    for pre_op in pre_ops:
        if _is_dynamic_weight_op(pre_op):
            return
        same_prune_before_elementwise_add = _find_weight_ops(
            pre_op, graph, same_prune_before_elementwise_add)
    return same_prune_before_elementwise_add


def _is_output_weight_ops(op, graph):
    next_ops = sorted(graph.next_ops(op))
    for next_op in next_ops:
        if op == next_op:
            continue
        if _is_dynamic_weight_op(next_op):
            return False
        return _is_output_weight_ops(next_op, graph)
    return True


def if_is_bias(op, graph):
    pre_ops = sorted(graph.pre_ops(op))
    if 'conv' in pre_ops[0].type() and pre_ops[1].type() == "reshape2":
        if pre_ops[1].inputs('X')[0]._var.persistable == True:
            return True
    return False


def check_search_space(graph):
    """ Find the shortcut in the model and set same config for this situation.
    """
    output_conv = []
    same_search_space = []
    depthwise_conv = []
    fixed_by_input = []
    for op in graph.ops():
        # if there is no weight ops after this op,
        # this op can be seen as an output
        if _is_output_weight_ops(op, graph) and _is_dynamic_weight_op(op):
            for inp in op.all_inputs():
                if inp._var.persistable:
                    output_conv.append(inp._var.name)

        if op.type() == 'elementwise_add' or op.type() == 'elementwise_mul':
            inp1, inp2 = op.all_inputs()[0], op.all_inputs()[1]
            is_bias = if_is_bias(op, graph)
            if ((not inp1._var.persistable) and
                (not inp2._var.persistable)) and not is_bias:
                # if one of two vars comes from input,
                # then the two vars in this elementwise op should be all fixed
                if inp1.inputs() and inp2.inputs():
                    pre_fixed_op_1, pre_fixed_op_2 = [], []
                    pre_fixed_op_1 = _find_weight_ops(inp1.inputs()[0], graph,
                                                      pre_fixed_op_1)
                    pre_fixed_op_2 = _find_weight_ops(inp2.inputs()[0], graph,
                                                      pre_fixed_op_2)
                    if not pre_fixed_op_1:
                        fixed_by_input += pre_fixed_op_2
                    if not pre_fixed_op_2:
                        fixed_by_input += pre_fixed_op_1
                elif (not inp1.inputs() and
                      inp2.inputs()) or (inp1.inputs() and not inp2.inputs()):
                    pre_fixed_op = []
                    inputs = inp1.inputs(
                    ) if not inp2.inputs() else inp2.inputs()
                    pre_fixed_op = _find_weight_ops(inputs[0], graph,
                                                    pre_fixed_op)
                    fixed_by_input += pre_fixed_op

                pre_ele_op = _find_pre_elementwise_op(op, graph)
                if pre_ele_op != None:
                    same_search_space.append(pre_ele_op)

        if _is_depthwise(op):
            for inp in op.all_inputs():
                if inp._var.persistable:
                    depthwise_conv.append(inp._var.name)

    if len(same_search_space) == 0:
        return None, [], [], output_conv

    same_search_space = sorted([sorted(x) for x in same_search_space])
    final_search_space = []

    if len(same_search_space) >= 1:
        final_search_space = [same_search_space[0]]
        if len(same_search_space) > 1:
            for l in same_search_space[1:]:
                listset = set(l)
                merged = False
                for idx in range(len(final_search_space)):
                    rset = set(final_search_space[idx])
                    if len(listset & rset) != 0:
                        final_search_space[idx] = list(listset | rset)
                        merged = True
                        break
                if not merged:
                    final_search_space.append(l)
    final_search_space = sorted([sorted(x) for x in final_search_space])
    depthwise_conv = sorted(depthwise_conv)
    fixed_by_input = sorted(fixed_by_input)

    return (final_search_space, depthwise_conv, fixed_by_input, output_conv)


def broadcast_search_space(same_search_space, param2key, origin_config):
    """
    Inplace broadcast the origin_config according to the same search space. Such as: same_search_space = [['conv1_weight', 'conv3_weight']], param2key = {'conv1_weight': 'conv1.conv', 'conv3_weight': 'conv3.weight'}, origin_config= {'conv1.weight': {'channel': 10}, 'conv2.weight': {'channel': 20}}, the result after this function is origin_config={'conv1.weight': {'channel': 10}, 'conv2.weight': {'channel': 20}, 'conv3.weight': {'channel': 10}}

    Args:
        same_search_space(list<list>): broadcast according this list, each list in same_search_space means the channel must be consistent.
        param2key(dict): the name of layers corresponds to the name of parameter.
        origin_config(dict): the search space which can be searched.
    """
    for per_ss in same_search_space:
        for ss in per_ss[1:]:
            key = param2key[ss]
            pre_key = param2key[per_ss[0]]
            if key in origin_config:
                if 'expand_ratio' in origin_config[pre_key]:
                    origin_config[key].update({
                        'expand_ratio':
                        origin_config[pre_key]['expand_ratio']
                    })
                elif 'channel' in origin_config[pre_key]:
                    origin_config[key].update({
                        'channel':
                        origin_config[pre_key]['channel']
                    })
            else:
                # if the pre_key is removed from config for some reasons
                # such as it is fixed by hand or by elementwise op
                if pre_key in origin_config:
                    if 'expand_ratio' in origin_config[pre_key]:
                        origin_config[key] = {
                            'expand_ratio':
                            origin_config[pre_key]['expand_ratio']
                        }
                    elif 'channel' in origin_config[pre_key]:
                        origin_config[key] = {
                            'channel': origin_config[pre_key]['channel']
                        }
