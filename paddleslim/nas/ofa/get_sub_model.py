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

__all__ = ['get_prune_params_config', 'prune_params', 'check_ss']

WEIGHT_OP = ['conv2d', 'conv3d', 'conv1d', 'linear', 'embedding']
CONV_TYPES = [
    'conv2d', 'conv3d', 'conv1d', 'superconv2d', 'supergroupconv2d',
    'superdepthwiseconv2d'
]


def get_prune_params_config(graph, origin_model_config):
    param_config = {}
    precedor = None
    for op in graph.ops():
        ### TODO(ceci3):
        ### 1. fix config when this op is concat by graph.pre_ops(op)
        ### 2. add kernel_size in config
        ### 3. add channel in config
        for inp in op.all_inputs():
            n_ops = graph.next_ops(op)
            if inp._var.name in origin_model_config.keys():
                if 'expand_ratio' in origin_model_config[inp._var.name].keys():
                    tmp = origin_model_config[inp._var.name]['expand_ratio']
                    if len(inp._var.shape) > 1:
                        if inp._var.name in param_config.keys():
                            param_config[inp._var.name].append(tmp)
                        ### first op
                        else:
                            param_config[inp._var.name] = [precedor, tmp]
                    else:
                        param_config[inp._var.name] = [tmp]
                    precedor = tmp
                else:
                    precedor = None
            for n_op in n_ops:
                for next_inp in n_op.all_inputs():
                    if next_inp._var.persistable == True:
                        if next_inp._var.name in origin_model_config.keys():
                            if 'expand_ratio' in origin_model_config[
                                    next_inp._var.name].keys():
                                tmp = origin_model_config[next_inp._var.name][
                                    'expand_ratio']
                                pre = tmp if precedor is None else precedor
                                if len(next_inp._var.shape) > 1:
                                    param_config[next_inp._var.name] = [pre]
                                else:
                                    param_config[next_inp._var.name] = [tmp]
                            else:
                                if len(next_inp._var.
                                       shape) > 1 and precedor != None:
                                    param_config[
                                        next_inp._var.name] = [precedor, None]
                        else:
                            param_config[next_inp._var.name] = [precedor]

    return param_config


def prune_params(model, param_config, super_model_sd=None):
    for l_name, sublayer in model.named_sublayers():
        if isinstance(sublayer, BaseBlock):
            continue
        for p_name, param in sublayer.named_parameters(include_sublayers=False):
            name = l_name + '.' + p_name
            t_value = param.value().get_tensor()
            value = np.array(t_value).astype("float32")

            if super_model_sd != None:
                super_t_value = super_model_sd[name].value().get_tensor()
                super_value = np.array(super_t_value).astype("float32")

            if param.name in param_config.keys():
                if len(param_config[param.name]) > 1:
                    in_exp = param_config[param.name][0]
                    out_exp = param_config[param.name][1]
                    if sublayer.__class__.__name__.lower() in CONV_TYPES:
                        in_chn = int(value.shape[1]) if in_exp == None else int(
                            value.shape[1] * in_exp)
                        out_chn = int(value.shape[
                            0]) if out_exp == None else int(value.shape[0] *
                                                            out_exp)
                        prune_value = super_value[:out_chn, :in_chn, ...] \
                                         if super_model_sd != None else value[:out_chn, :in_chn, ...]
                    else:
                        in_chn = int(value.shape[0]) if in_exp == None else int(
                            value.shape[0] * in_exp)
                        out_chn = int(value.shape[
                            1]) if out_exp == None else int(value.shape[1] *
                                                            out_exp)
                        prune_value = super_value[:in_chn, :out_chn, ...] \
                                         if super_model_sd != None else value[:in_chn, :out_chn, ...]
                else:
                    out_chn = int(value.shape[0]) if param_config[param.name][
                        0] == None else int(value.shape[0] *
                                            param_config[param.name][0])
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


def _find_weight_ops(op, graph, weights):
    pre_ops = graph.pre_ops(op)
    for pre_op in pre_ops:
        if pre_op.type() in WEIGHT_OP:
            for inp in pre_op.all_inputs():
                if inp._var.persistable:
                    weights.append(inp._var.name)
            return weights
        return _find_weight_ops(pre_op, graph, weights)


def _find_pre_elementwise_add(op, graph):
    same_ss_per_op = []
    pre_ops = graph.pre_ops(op)
    for pre_op in pre_ops:
        if pre_op.type() in WEIGHT_OP:
            return
        same_ss_per_op = _find_weight_ops(pre_op, graph, same_ss_per_op)
    return same_ss_per_op


def check_ss(graph):
    same_ss = []
    for op in graph.ops():
        if op.type() == 'elementwise_add':
            inp1, inp2 = op.all_inputs()[0], op.all_inputs()[1]
            if (not inp1._var.persistable) and (not inp2._var.persistable):
                same_ss.append(_find_pre_elementwise_add(op, graph))

    same_ss = sorted([sorted(x) for x in same_ss])
    if len(same_ss) == 0:
        return None
    final_ss = []

    if len(same_ss) >= 1:
        final_ss = [same_ss[0]]
        if len(same_ss) > 1:
            for l in same_ss[1:]:
                listset = set(l)
                merged = False
                for idx in range(len(final_ss)):
                    rset = set(final_ss[idx])
                    if len(listset & rset) != 0:
                        final_ss[idx] = list(listset | rset)
                        merged = True
                        break
                if not merged:
                    final_ss.append(l)

    return final_ss
