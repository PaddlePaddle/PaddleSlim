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
from paddle.fluid import core
from .layers_base import BaseBlock

__all__ = ['get_prune_params_config', 'prune_params', 'check_search_space']

### NOTE: the difference between DYNAMIC_WEIGHT_OP and ALL_WEIGHT_OP is whether 
### the shape of weight can changed proactive in forward.

DYNAMIC_WEIGHT_OP = [
    'conv2d', 'mul', 'matmul', 'embedding', 'conv2d_transpose',
    'depthwise_conv2d'
]
CONV_TYPES = [
    'conv2d', 'conv3d', 'conv1d', 'superconv2d', 'supergroupconv2d',
    'superdepthwiseconv2d'
]

ALL_WEIGHT_OP = [
    'conv2d', 'mul', 'matmul', 'elementwise_add', 'embedding',
    'conv2d_transpose', 'depthwise_conv2d', 'batch_norm', 'layer_norm',
    'instance_norm', 'sync_batch_norm'
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


def _get_precedor_of_concat(op, graph, origin_model_config):
    weights = []
    precedor = []
    for inp in sorted(op.all_inputs()):
        weight = []
        pre_op = inp.inputs()
        if len(pre_op) == 0:
            weights = weights + [None]
            continue
        weight = _find_weight_ops(pre_op[0], graph, weight, mode='pre')
        if weight == []:
            weights = weights + [None]
        else:
            weights = weights + weight
    for weight in weights:
        if weight is None:
            precedor.append(1.0)
            continue
        if weight.name in origin_model_config.keys():
            if 'expand_ratio' in origin_model_config[
                    weight.name] or 'channel' in origin_model_config[
                        weight.name]:
                key = 'channel' if 'channel' in origin_model_config[
                    weight.name] else 'expand_ratio'
                precedor.append(origin_model_config[weight.name][key])
        else:
            precedor.append(1.0)
    sum_prune_shape = 0
    sum_inputs_shape = 0
    if len(weights) != 0:
        for idx, inp in enumerate(sorted(op.all_inputs())):
            sum_prune_shape += inp.shape()[1] * precedor[idx]
            sum_inputs_shape += inp.shape()[1]
        return float(sum_prune_shape) / float(sum_inputs_shape)

    else:
        return None


def get_prune_params_config(graph, origin_model_config):
    """ Convert config of search space to parameters' prune config.
    """
    param_config = {}
    precedor = None
    for op in graph.ops():
        ### TODO(ceci3): add kernel_size in config
        ### if axis of concat is not 1, treat it as normal op.
        ### NOTE: only support data_format = 'NCHW' now.
        if op.type() == 'concat' and int(op.attr('axis')) == 1:
            precedor = _get_precedor_of_concat(op, graph, origin_model_config)
        else:
            _find_ofa_layers(op, graph)
            for inp in sorted(op.all_inputs()):
                if inp._var.name in origin_model_config.keys():
                    if 'expand_ratio' in origin_model_config[
                            inp._var.name] or 'channel' in origin_model_config[
                                inp._var.name]:
                        key = 'channel' if 'channel' in origin_model_config[
                            inp._var.name] else 'expand_ratio'
                        tmp = origin_model_config[inp._var.name][key]
                        if len(inp._var.shape) == 1:
                            param_config[inp._var.name] = [tmp]
                        precedor = tmp
                    else:
                        precedor = None
        ### find all next ops:
        ###   a. if next op with weight, the prune ratio of input channel in the 
        ###      next op is equal to the current op.
        ###   b. if next op without weight, find all the next op with weight of the next op by dfs 
        n_ops = sorted(graph.next_ops(op))
        for n_op in n_ops:
            _find_ofa_layers(n_op, graph)
            if _is_dynamic_weight_op(n_op):
                for next_inp in sorted(n_op.all_inputs()):
                    if next_inp._var.persistable == True:
                        next_inp = _clear_ofa_layers(next_inp)
                        ### the key of *_norm will not in origin_model_config
                        ### so if n_op is *_norm, will pass to else branch certainly.
                        if next_inp._var.name in origin_model_config.keys():
                            if 'expand_ratio' in origin_model_config[
                                    next_inp._var.
                                    name] or 'channel' in origin_model_config[
                                        next_inp._var.name]:
                                key = 'channel' if 'channel' in origin_model_config[
                                    next_inp._var.name] else 'expand_ratio'
                                tmp = origin_model_config[next_inp._var.name][
                                    key]
                                if len(next_inp._var.shape) > 1:
                                    param_config[
                                        next_inp._var.name] = [precedor, tmp]
                            else:
                                if len(next_inp._var.
                                       shape) > 1 and precedor != None:
                                    param_config[
                                        next_inp._var.name] = [precedor, None]
            else:
                weights = []
                _find_weight_ops(
                    n_op, graph, weights, mode='next', all_weight_op=True)
                for var in weights:
                    if var is None:
                        continue
                    if var.name not in origin_model_config and var.name not in param_config:
                        if len(var.shape) > 1:
                            param_config[var.name] = [precedor, None]
                        else:
                            param_config[var.name] = [precedor]
    return param_config


def _find_ofa_layers(op, graph):
    ### find slice op add by ofa layers and set the 
    ### output.persistable = True if input.persistable = True
    for pre_op in sorted(graph.pre_ops(op)):
        if pre_op.type() == 'slice' and op.type() in ALL_WEIGHT_OP:
            ### slice op has only one input and one output
            if sorted(pre_op.all_inputs())[0]._var.persistable == True:
                sorted(pre_op.all_outputs())[0]._var.persistable = True


def _clear_ofa_layers(inp):
    pre_op = inp.inputs()
    if len(pre_op) != 0 and pre_op[0].type() == 'slice':
        pre_inp = pre_op[0].all_inputs()[0]
        return pre_inp
    else:
        return inp


def _find_weight_ops(op, graph, weights, mode='pre', all_weight_op=False):
    """ Find the vars come from operators with weight.
    """
    if mode == 'pre':
        find_ops = sorted(graph.pre_ops(op))
    elif mode == 'next':
        find_ops = sorted(graph.next_ops(op))
    else:
        raise NotImplementedError(
            "there is something wrong in parameter \'mode\', \'mode\' must in ['pre', 'next'], but now is {}".
            format(mode))
    for f_op in find_ops:
        find_weight_op = False
        ### if op == 'batch_norm', pre_ops of batch_norm will have itself.
        ### because the mean and variance is the inputs and outputs at the same time.
        if f_op == op:
            continue
        _find_ofa_layers(f_op, graph)
        ### if depthwise conv is one of elementwise's input, 
        ### add it into this same search space
        if _is_depthwise(f_op):
            for inp in sorted(f_op.all_inputs()):
                if inp._var.persistable:
                    inp = _clear_ofa_layers(inp)
                    weights.append(inp._var)

        if _is_dynamic_weight_op(f_op,
                                 all_weight_op) and not _is_depthwise(f_op):
            for inp in sorted(f_op.all_inputs()):
                if inp._var.persistable:
                    if _is_dynamic_weight_op(f_op):
                        find_weight_op = True
                    inp = _clear_ofa_layers(inp)
                    weights.append(inp._var)
            return weights
        if find_weight_op == False:
            _find_weight_ops(f_op, graph, weights)
    return weights


def get_actual_shape(transform, channel):
    if transform == None:
        channel = int(channel)
    else:
        if isinstance(transform, float):
            channel = int(channel * transform)
        else:
            channel = int(transform)
    return channel


def prune_params(model, param_config, super_model_sd=None):
    """ Prune parameters according to the config.
        Parameters:
            model(paddle.nn.Layer): instance of model.
            param_config(dict): prune config of each weight.
            super_model_sd(dict, optional): parameters come from supernet. If super_model_sd is not None, transfer parameters from this dict to model; otherwise, prune model from itself.
    """
    for l_name, sublayer in model.named_sublayers():
        if isinstance(sublayer, BaseBlock):
            continue
        for p_name, param in sublayer.named_parameters(include_sublayers=False):
            t_value = param.value().get_tensor()
            value = np.array(t_value).astype("float32")

            if super_model_sd != None:
                name = l_name + '.' + p_name
                super_t_value = super_model_sd[name].value().get_tensor()
                super_value = np.array(super_t_value).astype("float32")
                super_model_sd.pop(name)

            if param.name in param_config.keys():
                if len(param_config[param.name]) > 1:
                    in_exp = param_config[param.name][0]
                    out_exp = param_config[param.name][1]
                    if sublayer.__class__.__name__.lower() in CONV_TYPES:
                        in_chn = get_actual_shape(in_exp, value.shape[1])
                        out_chn = get_actual_shape(out_exp, value.shape[0])
                        prune_value = super_value[:out_chn, :in_chn, ...] \
                                         if super_model_sd != None else value[:out_chn, :in_chn, ...]
                    else:
                        in_chn = get_actual_shape(in_exp, value.shape[0])
                        out_chn = get_actual_shape(out_exp, value.shape[1])
                        prune_value = super_value[:in_chn, :out_chn, ...] \
                                         if super_model_sd != None else value[:in_chn, :out_chn, ...]
                else:
                    out_chn = get_actual_shape(param_config[param.name][0],
                                               value.shape[0])
                    prune_value = super_value[:out_chn, ...] \
                                     if super_model_sd != None else value[:out_chn, ...]

            else:
                prune_value = super_value if super_model_sd != None else value

            p = t_value._place()
            if p.is_cpu_place():
                place = core.CPUPlace()
            elif p.is_cuda_pinned_place():
                place = core.CUDAPinnedPlace()
            else:
                place = core.CUDAPlace(p.gpu_device_id())
            t_value.set(prune_value, place)
            if param.trainable:
                param.clear_gradient()

    ### initialize param which not in sublayers, such as create persistable inputs by create_parameters
    if super_model_sd != None and len(super_model_sd) != 0:
        for k, v in super_model_sd.items():
            setattr(model, k, v)


def _is_depthwise(op):
    """Check if this op is depthwise conv. Only Cin == Cout == groups be consider as depthwise conv.
       The shape of input and the shape of output in depthwise conv must be same in superlayer,
       so depthwise op cannot be consider as weight op
    """
    #if op.type() == 'depthwise_conv2d': ### depthwise_conv2d in paddle is Cout % Cin =0
    #    return True
    if 'conv' in op.type():
        for inp in sorted(op.all_inputs()):
            if inp._var.persistable and (
                    op.attr('groups') == inp._var.shape[0] and
                    op.attr('groups') * inp._var.shape[1] == inp._var.shape[0]):
                return True
    return False


def _find_pre_elementwise_add(op, graph):
    """ Find precedors of the elementwise_add operator in the model.
    """
    same_prune_before_elementwise_add = []
    pre_ops = sorted(graph.pre_ops(op))
    for pre_op in pre_ops:
        if _is_dynamic_weight_op(pre_op):
            return
        same_prune_before_elementwise_add = _find_weight_ops(
            pre_op, graph, same_prune_before_elementwise_add, mode='pre')
        new_same_prune_before_elementwise_add = []
        for key in same_prune_before_elementwise_add:
            if key is not None:
                new_same_prune_before_elementwise_add.append(key.name)
    return new_same_prune_before_elementwise_add


def check_search_space(graph):
    """ Find the shortcut in the model and set same config for this situation.
    """
    same_search_space = []
    depthwise_conv = []
    pre_reshape_dynamic_weight_op = set()
    tmp_op = []
    for op in graph.ops():
        ### if current op is reshape, and all dim in the shape cannot change,
        ### the output channel of precedor dynamic op of this op cannot change too.
        if op.type() == 'reshape2':
            find_unknown = False
            for shape in op.attr('shape')[1:]:
                if shape == -1:
                    find_unknown = True
            if find_unknown == False:
                tmp_op = _find_weight_ops(op, graph, tmp_op, mode='pre')
                for t_op in tmp_op:
                    if t_op is not None:
                        pre_reshape_dynamic_weight_op.add(t_op.name)
                tmp_op = []
        if op.type() == 'elementwise_add' or op.type() == 'elementwise_mul':
            inp1, inp2 = sorted(op.all_inputs())[0], sorted(op.all_inputs())[1]
            if (not inp1._var.persistable) and (not inp2._var.persistable):
                pre_ele_op = _find_pre_elementwise_add(op, graph)
                if pre_ele_op != None:
                    same_search_space.append(pre_ele_op)
        if _is_depthwise(op):
            for inp in sorted(op.all_inputs()):
                if inp._var.persistable:
                    depthwise_conv.append(inp._var.name)

    pre_reshape_dynamic_weight_op = sorted(pre_reshape_dynamic_weight_op)

    if len(same_search_space) == 0:
        return None, [], pre_reshape_dynamic_weight_op

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
    ### if there is the output channel cannot be changed in same search space
    for tmp_ss in final_search_space:
        cannot_change = False
        for s in tmp_ss:
            if s in pre_reshape_dynamic_weight_op:
                cannot_change = True
                pre_reshape_dynamic_weight_op.append(s)
        if cannot_change == True:
            final_search_space.remove(tmp_ss)
    final_search_space = sorted([sorted(x) for x in final_search_space])
    depthwise_conv = sorted(depthwise_conv)
    pre_reshape_dynamic_weight_op = sorted(set(pre_reshape_dynamic_weight_op))
    return (final_search_space, depthwise_conv, pre_reshape_dynamic_weight_op)


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
                        'expand_ratio': origin_config[pre_key]['expand_ratio']
                    })
                elif 'channel' in origin_config[pre_key]:
                    origin_config[key].update({
                        'channel': origin_config[pre_key]['channel']
                    })
            else:
                if 'expand_ratio' in origin_config[pre_key]:
                    origin_config[key] = {
                        'expand_ratio': origin_config[pre_key]['expand_ratio']
                    }
                elif 'channel' in origin_config[pre_key]:
                    origin_config[key] = {
                        'channel': origin_config[pre_key]['channel']
                    }
