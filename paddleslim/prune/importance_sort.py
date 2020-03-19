"""Define some functions to sort substructures of parameter by importance.
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
from ..core import GraphWrapper
from ..common import get_logger

__all__ = ["channel_score_sort", "batch_norm_scale_sort"]


def channel_score_sort(group, graph):
    """Sort channels of convolution by importance.

    This function return a list of parameters' sorted indexes on given axis.
    Each element of list is a tuple with format (name, axis, indexes) in which 'name' is parameter's name
    and 'axis' is the axis pruning on and `indexes` is sorted indexes.

    The sorted indexes is computed by below steps:

    step1: Find the first convolution layer in given group.
    step2: Get the scores of first convolution's channels.
    step3: Get sorted indexes by calling scores.argsort().
    step4: All the parameters in group share the same sorted indexes computed in step3.

    Args:
       group(list): A group of parameters. The first parameter of the group is convolution layer's weight
                    while the others are parameters affected by pruning the first one. Each parameter in group
                    is represented as tuple '(name, axis, score)' in which `name` is the parameter's name and
                    `axis` is the axis pruning on and `score` is a np.array storing the importance of strucure
                    on `axis`. Show as below:

                    .. code-block: text

                       [("conv1_weights", 0, [0.7, 0.5, 0.6]), ("conv1_bn.scale", 0, [0.1, 0.2, 0.4])]

                    The shape of "conv1_weights" is `[out_channel, in_channel, filter_size, filter_size]`, so
                    `[0.7, 0.5, 0.6]` are the importance sores of each output channel in "conv1_weights"
                    while axis is 0.
     

       graph(GraphWrapper): The graph is an auxiliary for sorting. It won't be used in this function.

    Returns:

       list: sorted indexes

    """
    name, axis, score = group[
        0]  # sort channels by the first convolution's score
    sorted_idx = score.argsort()
    idxs = []
    for name, axis, score in group:
        idxs.append((name, axis, sorted_idx))
    return idxs


def batch_norm_scale_sort(group, graph):
    """Sort channels of convolution by scales in batch norm layer.

    This function return a list of parameters' sorted indexes on given axis.
    Each element of list is a tuple with format (name, axis, indexes) in which 'name' is parameter's name
    and 'axis' is the axis pruning on and `indexes` is sorted indexes.

    The sorted indexes is computed by below steps:

    step1: Find the batch norm layer after the first convolution in given group.
    step2: Get the scales of the batch norm layer.
    step3: Get sorted indexes by calling `scales.argsort()`.
    step4: All the parameters in group share the same sorted indexes computed in step3.

    Args:
       group(list): A group of parameters. The first parameter of the group is convolution layer's weight
                    while the others are parameters affected by pruning the first one. Each parameter in group
                    is represented as tuple '(name, axis, score)' in which `name` is the parameter's name and
                    `axis` is the axis pruning on and `score` is a np.array storing the importance of strucure
                    on `axis`. Show as below:

                    .. code-block: text

                       [("conv1_weights", 0, [0.7, 0.5, 0.6]), ("conv1_bn.scale", 0, [0.1, 0.2, 0.4])]

                    The shape of "conv1_weights" is `[out_channel, in_channel, filter_size, filter_size]`, so
                    `[0.7, 0.5, 0.6]` are the importance sores of each output channel in "conv1_weights"
                    while axis is 0.
     

       graph(GraphWrapper): The graph is an auxiliary for sorting. It is used to find
                            the batch norm layer after given convolution layer.

    Returns:
       list: sorted indexes
    """
    assert (isinstance(graph, GraphWrapper))
    # step1: Get first convolution
    conv_weight, axis, score = group[0]
    param_var = graph.var(conv_weight)
    conv_op = param_var.outputs()[0]

    # step2: Get bn layer after first convolution
    conv_output = conv_op.outputs("Output")[0]
    bn_op = conv_output.outputs()[0]
    if bn_op is not None:
        bn_scale_param = bn_op.inputs("Scale")[0].name()
    else:
        raise SystemExit("Can't find BatchNorm op after Conv op in Network.")

    # steps3: Find score of bn and compute sorted indexes 
    sorted_idx = None
    for name, axis, score in group:
        if name == bn_scale_param:
            sorted_idx = score.argsort()
            break

    # step4: Share the sorted indexes with all the parameter in group
    idxs = []
    if sorted_idx is not None:
        for name, axis, score in group:
            idxs.append((name, axis, sorted_idx))
    return idxs
