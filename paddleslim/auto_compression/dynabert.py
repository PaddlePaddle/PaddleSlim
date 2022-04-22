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


def update_stack_inputs(program, op, new_inputs_len):
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
