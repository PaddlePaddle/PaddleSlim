"""Define some functions to compute the importance of structure to be pruned.
"""
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import numpy as np
from ..common import get_logger
from ..core import Registry, GraphWrapper

__all__ = ["l1_norm", "CRITERION"]

_logger = get_logger(__name__, level=logging.INFO)

CRITERION = Registry('criterion')


@CRITERION.register
def l1_norm(group, graph):
    """Compute l1-norm scores of parameter on given axis.

    This function return a list of parameters' l1-norm scores on given axis.
    Each element of list is a tuple with format (name, axis, score) in which 'name' is parameter's name
    and 'axis' is the axis reducing on and `score` is a np.array storing the l1-norm of strucure on `axis`.

    Args:
       group(list): A group of parameters. The first parameter of the group is convolution layer's weight
                    while the others are parameters affected by pruning the first one. Each parameter in group
                    is represented as tuple '(name, values, axis)' in which `name` is the parameter's name and
                    and `values` is the values of parameter and `axis` is the axis reducing on pruning on.
    Returns:
       list: A list of tuple storing l1-norm on given axis.
    """
    scores = []
    for name, value, axis, pruned_idx in group:

        reduce_dims = [i for i in range(len(value.shape)) if i != axis]
        score = np.sum(np.abs(value), axis=tuple(reduce_dims))
        scores.append((name, axis, score, pruned_idx))

    return scores


@CRITERION.register
def geometry_median(group, graph):
    scores = []
    name, value, axis, _ = group[0]
    assert (len(value.shape) == 4)

    def get_distance_sum(value, out_idx):
        w = value.view()
        w.shape = value.shape[0], np.product(value.shape[1:])
        selected_filter = np.tile(w[out_idx], (w.shape[0], 1))
        x = w - selected_filter
        x = np.sqrt(np.sum(x * x, -1))
        return x.sum()

    dist_sum_list = []
    for out_i in range(value.shape[0]):
        dist_sum = get_distance_sum(value, out_i)
        dist_sum_list.append(dist_sum)

    tmp = np.array(dist_sum_list)

    for name, value, axis, idx in group:
        scores.append((name, axis, tmp, idx))
    return scores


@CRITERION.register
def bn_scale(group, graph):
    """Compute l1-norm scores of parameter on given axis.

    This function return a list of parameters' l1-norm scores on given axis.
    Each element of list is a tuple with format (name, axis, score) in which 'name' is parameter's name
    and 'axis' is the axis reducing on and `score` is a np.array storing the l1-norm of strucure on `axis`.

    Args:
       group(list): A group of parameters. The first parameter of the group is convolution layer's weight
                    while the others are parameters affected by pruning the first one. Each parameter in group
                    is represented as tuple '(name, values, axis)' in which `name` is the parameter's name and
                    and `values` is the values of parameter and `axis` is the axis reducing on pruning on.
    Returns:
       list: A list of tuple storing l1-norm on given axis.
    """
    assert (isinstance(graph, GraphWrapper))

    # step1: Get first convolution
    conv_weight, value, axis, _ = group[0]
    param_var = graph.var(conv_weight)
    conv_op = param_var.outputs()[0]

    # step2: Get bn layer after first convolution
    conv_output = conv_op.outputs("Output")[0]
    bn_op = conv_output.outputs()[0]
    if bn_op is not None:
        bn_scale_param = bn_op.inputs("Scale")[0].name()
    else:
        raise SystemExit("Can't find BatchNorm op after Conv op in Network.")

    # steps3: Find scale of bn
    score = None
    for name, value, aixs, _ in group:
        if bn_scale_param == name:
            score = np.abs(value.reshape([-1]))

    scores = []
    for name, value, axis, idx in group:
        scores.append((name, axis, score, idx))

    return scores
