# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import numpy as np
from paddle.fluid.framework import IrNode
from paddle.fluid.framework import Operator

_op_real_in_out_name = {
    "conv2d": [["Input", "Filter"], ["Output"]],
    "depthwise_conv2d": [["Input", "Filter"], ["Output"]],
    "conv2d_transpose": [["Input", "Filter"], ["Output"]],
    "mul": [["X", "Y"], ["Out"]],
    "matmul": [["X", "Y"], ["Out"]],
    "matmul_v2": [["X", "Y"], ["Out"]],
    "pool2d": [["X"], ["Out"]],
    "elementwise_add": [["X", "Y"], ["Out"]],
    "concat": [["X"], ["Out"]],
    "softmax": [["X"], ["Out"]],
    "argmax": [["X"], ["Out"]],
    "transpose": [["X"], ["Out"]],
    "equal": [["X", "Y"], ["Out"]],
    "gather": [["X"], ["Out"]],
    "greater_equal": [["X", "Y"], ["Out"]],
    "greater_than": [["X", "Y"], ["Out"]],
    "less_equal": [["X", "Y"], ["Out"]],
    "less_than": [["X", "Y"], ["Out"]],
    "mean": [["X"], ["Out"]],
    "not_equal": [["X", "Y"], ["Out"]],
    "reshape": [["X"], ["Out"]],
    "reshape2": [["X"], ["Out"]],
    "transpose2": [["X"], ["Out"]],
    "bilinear_interp": [["X"], ["Out"]],
    "nearest_interp": [["X"], ["Out"]],
    "trilinear_interp": [["X"], ["Out"]],
    "slice": [["Input"], ["Out"]],
    "squeeze": [["X"], ["Out"]],
    "elementwise_sub": [["X", "Y"], ["Out"]],
    "relu": [["X"], ["Out"]],
    "relu6": [["X"], ["Out"]],
    "leaky_relu": [["X"], ["Out"]],
    "prelu": [["X", "Alpha"], ["Out"]],
    "tanh": [["X"], ["Out"]],
    "swish": [["X"], ["Out"]],
    "dropout": [["X"], ["Out"]],
    "batch_norm": [["X"], ["Y"]],
    "layer_norm": [["X"], ["Y"]],
    "sigmoid": [["X"], ["Out"]],
    "elementwise_mul": [["X", "Y"], ["Out"]],
    "elementwise_pow": [["X", "Y"], ["Out"]],
    "hard_swish": [["X"], ["Out"]],
    "hard_sigmoid": [["X"], ["Out"]],
    "gru": [["Input", "Weight"], ["Hidden"]],
    "lstm": [["Input", "Weight"], ["Hidden"]],
    "pad2d": [["X"], ["Out"]],
    "pad3d": [["X"], ["Out"]],
    "flatten": [["X"], ["Out"]],
    "flatten2": [["X"], ["Out"]],
    "unsqueeze2": [["X"], ["Out"]],
    "unsqueeze2": [["X"], ["Out"]],
    "flatten_contiguous_range": [["X"], ["Out"]],
    "split": [["X"], ["Out"]],
    "squeeze2": [["X"], ["Out"]],
    "nearest_interp_v2": [["X"], ["Out"]],
    "bilinear_interp": [["X"], ["Out"]],
    "bilinear_interp_v2": [["X"], ["Out"]],
    "fill_constant_batch_size_like": [["Input"], ["Out"]],
    "arg_max": [["X"], ["Out"]],
    "abs": [["X"], ["Out"]],
    "assign": [["X"], ["Out"]],
    "cast": [["X"], ["Out"]],
    "clip": [["X"], ["Out"]],
    "box_coder": [["PriorBox"], ["OutputBox"]],
    "crop": [["X"], ["Out"]],
    "cumsum": [["X"], ["Out"]],
    "expand_v2": [["X"], ["Out"]],
    "fill_any_like": [["X"], ["Out"]],
    "fill_constant": [[], ["Out"]],
    "gelu": [["X"], ["Out"]],
    "instance_norm": [["X"], ["Out"]],
    "lookup_table": [["W", "Ids"], ["Out"]],
    "lookup_table_v2": [["W", "Ids"], ["Out"]],
    "norm": [["X"], ["Norm"]],
    "p_norm": [["X"], ["Out"]],
    "pow": [["X"], ["Out"]],
    "reduce_mean": [["X"], ["Out"]],
    "stack": [["X"], ["Y"]],
    "top_k_v2": [["X"], ["Out", "Indices"]],
    "logical_and": [["X", "Y"], ["Out"]],
    "logical_not": [["X"], ["Out"]],
    "meshgrid": [["X"], ["Out"]],
    "roi_align": [["X", "ROIs"], ["Out"]],
    "strided_slice": [["Input"], ["Out"]],
    "where": [["Condition", "X", "Y"], ["Out"]],
    "grid_sampler": [["X", "Grid"], ["Output"]],
    "tile": [["X"], ["Out"]],
    "group_norm": [["X"], ["Y", "Mean", "Variance"]],
    "reduce_sum": [["X"], ["Out"]],
    "square": [["X"], ["Out"]],
    "softplus": [["X"], ["Out"]],
    "shuffle_channel": [["X"], ["Out"]],
}


def _valid_format(data):
    is_dict = isinstance(data, dict)
    list_with_one_dict = isinstance(
        data, list) and len(data) == 1 and isinstance(data[0], dict)
    return is_dict or list_with_one_dict


def wrap_dataloader(dataloader, names):
    """Create a wrapper of dataloader if the data returned by the dataloader is not a dict.
    And the names will be the keys of dict returned by the wrapper.
    """
    if dataloader is None:
        return dataloader
    data = next(dataloader())
    if _valid_format(data):
        return dataloader

    if isinstance(data, Iterable):
        assert len(data) == len(
            names
        ), f"len(data) == len(names), but got len(data): {len(data)} and len(names): {len(names)}"
    else:
        assert len(
            names
        ) == 1, f"The length of name should 1 when data is not Iterable but got {len(names)}"

    def gen():
        for i, data in enumerate(dataloader()):
            if not isinstance(data, Iterable):
                data = [data]
            yield dict((name_, np.array(data_))
                       for name_, data_ in zip(names, data))

    return gen


def _get_op_input_var_names(op):
    """
    Get the input var names of the op.
    Args:
        op(IrNode, Operator): the input op.
    Returns:
        input_var_names or None.
    """
    assert isinstance(op, (IrNode, Operator)), \
        "The input op should be IrNode or Operator."
    var_names = []
    op_name = op.name() if isinstance(op, IrNode) \
        else op.type
    if op_name not in _op_real_in_out_name:
        return []

    name_list = _op_real_in_out_name[op_name][0]
    for name in name_list:
        var_name = op.input(name)
        if isinstance(var_name, list):
            var_names.extend(var_name)
        else:
            var_names.append(var_name)
    return var_names


def load_variable_data(scope, var_name):
    '''
    Load variable value from scope
    '''
    var_node = scope.find_var(var_name)
    assert var_node is not None, \
        "Cannot find " + var_name + " in scope."
    return np.array(var_node.get_tensor())
