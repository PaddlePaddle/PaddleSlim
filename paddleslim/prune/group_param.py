"""Define some functions to collect ralated parameters into groups."""
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
from .prune_walker import PRUNE_WORKER

__all__ = ["collect_convs"]

_logger = get_logger(__name__, level=logging.INFO)


def collect_convs(params, graph, visited={}):
    """Collect convolution layers of graph into groups. The layers in the same group is relative on pruning operation.
    A group is a list of tuple with format (param_name, axis) in which `param_name` is the name of parameter and `axis` is the axis to be pruned on.

    .. code-block:: text

       conv1->conv2->conv3->conv4

    As shown above, the demo has 4 convolution layers. And the shape of convolution's parameter is `[out_channel, in_channel, filter_size, filter_size]`. If parameter of `conv1` was pruned on axis 0, then the parameter of `conv2` should be pruned on axis 1. So the `conv1` and `conv2` is a group that can be represented as:

    .. code-block:: python

       [("conv1", 0), ("conv2", 1)]

    If `params` is `["conv1", "conv2"]`, then the returned groups is:

    .. code-block:: python

       [[("conv1", 0), ("conv2", 1)],
        [("conv2", 0), ("conv3", 1)]]

    Args:
       params(list): A list of convolution layer's parameter names. It will collect all the groups that contains anyone of these parameters.
       graph(paddle.static.Program | GraphWrapper): The graph used to search the groups.

    Returns:
       list<list<tuple>>: The groups.

    """
    if not isinstance(graph, GraphWrapper):
        graph = GraphWrapper(graph)
    groups = []
    for _param in params:
        pruned_params = []
        param = graph.var(_param)
        if param is None:
            _logger.warning(
                "Cann't found relative variables of {} because {} is not in target program or model. Please make sure {} is in your program if you are using static API of PaddlePaddle. And make sure your model in correctly mode and contains {} if you are using dynamic API of PaddlePaddle.".format(_param, _param, _param, _param)
            )
            groups.append([])
            continue
        target_op = param.outputs()[0]
        if target_op.type() == 'conditional_block':
            for op in param.outputs():
                if op.type() in PRUNE_WORKER._module_dict.keys():
                    cls = PRUNE_WORKER.get(op.type())
                    walker = cls(op,
                                 pruned_params=pruned_params,
                                 visited=visited)
                    break
        else:
            cls = PRUNE_WORKER.get(target_op.type())
            if cls is None:
                _logger.info("No walker for operator: {}".format(target_op.type(
                )))
                groups.append(pruned_params)
                continue
            walker = cls(target_op,
                         pruned_params=pruned_params,
                         visited=visited)

        walker.prune(param, pruned_axis=0, pruned_idx=[])
        groups.append(pruned_params)
    visited = set()
    uniq_groups = []
    for group in groups:
        repeat_group = False
        simple_group = []
        for param, axis, pruned_idx in group:
            param = param.name()
            if axis == 0:
                if param in visited:
                    repeat_group = True
                else:
                    visited.add(param)
            simple_group.append((param, axis, pruned_idx))
        if not repeat_group:
            uniq_groups.append(simple_group)
    return uniq_groups
