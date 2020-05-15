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

from ..core import GraphWrapper
from .prune_walker import conv2d as conv2d_walker

__all__ = ["collect_convs"]


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
       graph(paddle.fluid.Program | GraphWrapper): The graph used to search the groups.

    Returns:
       list<list<tuple>>: The groups.

    """
    if not isinstance(graph, GraphWrapper):
        graph = GraphWrapper(graph)
    groups = []
    for param in params:
        pruned_params = []
        param = graph.var(param)
        conv_op = param.outputs()[0]
        walker = conv2d_walker(
            conv_op, pruned_params=pruned_params, visited=visited)
        walker.prune(param, pruned_axis=0, pruned_idx=[])
        groups.append(pruned_params)
    visited = set()
    uniq_groups = []
    for group in groups:
        repeat_group = False
        simple_group = []
        for param, axis, _ in group:
            param = param.name()
            if axis == 0:
                if param in visited:
                    repeat_group = True
                else:
                    visited.add(param)
            simple_group.append((param, axis))
        if not repeat_group:
            uniq_groups.append(simple_group)

    return uniq_groups
