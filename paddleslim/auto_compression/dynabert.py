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

import numpy as np


def update_input_mask_inputs(program, op, new_inputs_len):
    input_var_name = op.input_arg_names
    block = program.blocks[0]
    var = block.var(input_var_name[0])
    op.desc.set_input('X', input_var_name[:int(len(input_var_name) * 0.5)])


def prune_weight(graph, scope, place, pruned_name, pruned_ratio):
    param = graph.var(pruned_name)
    _var = scope.find_var(param.name())
    if _var is None:
        return
    param_t = _var.get_tensor()
    pruned_ratio = [pruned_ratio[1]] if len(param_t.shape(
    )) == 1 else pruned_ratio
    print(pruned_name, param_t.shape(), pruned_ratio)
    pruned_shape = np.multiply(param_t.shape(), pruned_ratio)
    pruned_shape = list(map(int, pruned_shape))
    param.set_shape(pruned_shape)
    if len(pruned_shape) == 2:
        pruned_param = np.array(param_t)[:pruned_shape[0], :pruned_shape[1]]
    else:
        pruned_param = np.array(param_t)[:pruned_shape[0]]
    param_t.set(pruned_param, place)


def prune_transformer(scope, place, graph, pruned_dict):
    for name, value in pruned_dict.items():
        ### prune weight
        prune_weight(graph, scope, place, name, value)
    graph.infer_shape()
    return graph.program


def reorder_head(scope, place, weight, head_num, idx):
    qkv = weight['P1']
    attn_out = weight['P2']
    attn_out_t = scope.find_var(qkv[0]).get_tensor()
    num_per_head = int(attn_out_t.shape()[0] / head_num)

    index = np.reshape(
        np.take(
            np.reshape(
                np.arange(
                    0, head_num * num_per_head, dtype='int64'),
                (head_num, num_per_head)),
            idx,
            axis=0), (-1))

    def reorder_head_matrix(w_name, index, dim):
        pd_w = scope.find_var(w_name).get_tensor()
        np_w = np.array(pd_w)

        new_w = np.take(np_w, index, axis=dim)
        pd_w.set(new_w, place)

    for w_idx, weight_name in enumerate(qkv):
        if w_idx % 2 == 0:
            ### reorder qkv weight 
            reorder_head_matrix(weight_name, index, dim=1)
        else:
            ### reorder qkv bias 
            reorder_head_matrix(weight_name, index, dim=0)

    ### reorder attention output weight 
    reorder_head_matrix(attn_out[0], index, dim=0)


def reorder_neuron(scope, place, weight, idx):
    ffn_i = weight['P1']
    ffn_o = weight['P2']

    def reorder_neurons_matrix(w_name, index, dim):
        pd_w = scope.find_var(w_name).get_tensor()
        np_w = np.array(pd_w)

        new_w = np.take(np_w, index, axis=dim)
        pd_w.set(new_w, place)

    reorder_neurons_matrix(ffn_i[0], idx, dim=1)
    reorder_neurons_matrix(ffn_i[1], idx, dim=0)
    reorder_neurons_matrix(ffn_o[0], idx, dim=0)


def reorder_neuron_head(scope, place, mha_weight, ffn_weight, head_importance,
                        neuron_importance, head_num):
    for layer, current_importance in enumerate(neuron_importance):
        # reorder heads
        idx = np.argsort(head_importance[layer])[::-1]
        reorder_head(scope, place, mha_weight[layer], head_num, idx)
        #### reorder neurons
        idx = np.argsort(current_importance)[::-1]
        reorder_neuron(scope, place, ffn_weight[layer], idx)
