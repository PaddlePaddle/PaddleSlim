# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
import sys
import numpy as np
import paddle.fluid as fluid
import copy
from ..core import GraphWrapper
from ..core import DyGraph
from .group_param import collect_convs
from .criterion import l1_norm
from .importance_sort import channel_score_sort, batch_norm_scale_sort
from ..common import get_logger
import torch

__all__ = ["Pruner"]

_logger = get_logger(__name__, level=logging.INFO)


class Pruner():
    """The pruner used to prune channels of convolution.

    Args:
        criterion(str|function): the criterion used to sort channels for pruning.
        channel_sortor(str|function): 

    """

    def __init__(self, criterion="l1_norm", channel_sortor="channel_score"):
        self.criterion = criterion
        self.channel_sortor = channel_sortor
        if criterion == "l1_norm":
            self.criterion = l1_norm

        if channel_sortor == "channel_score":
            self.channel_sortor = channel_score_sort
        elif channel_sortor == "batch_norm_scale":
            self.channel_sortor = batch_norm_scale_sort

    def prune(self,
              graph,
              scope,
              params,
              ratios,
              place=None,
              lazy=False,
              only_graph=False,
              param_backup=False,
              param_shape_backup=False,
              input_shape=None):
        """Pruning the given parameters.

        Args:

            program(fluid.Program): The program to be pruned.
            scope(fluid.Scope): The scope storing paramaters to be pruned.
            params(list<str>): A list of parameter names to be pruned.
            ratios(list<float>): A list of ratios to be used to pruning parameters.
            place(fluid.Place): The device place of filter parameters. Defalut: None.
            lazy(bool): True means setting the pruned elements to zero.
                        False means cutting down the pruned elements. Default: False.
            only_graph(bool): True means only modifying the graph.
                              False means modifying graph and variables in scope. Default: False.
            param_backup(bool): Whether to return a dict to backup the values of parameters. Default: False.
            param_shape_backup(bool): Whether to return a dict to backup the shapes of parameters. Default: False.

        Returns:
            tuple: ``(pruned_program, param_backup, param_shape_backup)``. ``pruned_program`` is the pruned program. ``param_backup`` is a dict to backup the values of parameters. ``param_shape_backup`` is a dict to backup the shapes of parameters.
        """

        self.pruned_list = []
        if isinstance(graph, fluid.Program):
            graph = GraphWrapper(program.clone())
        elif isinstance(graph, torch.nn.Module):
            assert (
                input_shape is not None,
                "input_shape can not be None while graph is instance of torch.nn.Module"
            )
            graph = DyGraph(graph, input_shape)
        else:
            raise NotImplementedError('The type of graph is not supported.')
        param_backup = {} if param_backup else None
        param_shape_backup = {} if param_shape_backup else None

        visited = {}
        pruned_params = []
        for param, ratio in zip(params, ratios):
            group = collect_convs([param], graph)[0]  # [(name, axis)]
            print "group: {}".format(group)
            if only_graph:

                param_v = graph.var(param)
                pruned_num = int(round(param_v.shape()[0] * ratio))
                pruned_idx = [0] * pruned_num
                for name, aixs in group:
                    pruned_params.append((name, axis, pruned_idx))

            else:

                group_values = []
                for name, axis in group:
                    values = graph.var(name).data()
                    group_values.append((name, values, axis))

                scores = self.criterion(group_values)  # [(name, axis, score)]
                print "scores: {}".format(scores)
                group_idx = self.channel_sortor(
                    scores, graph=graph)  # [(name, axis, soted_idx)]
                print "group_idx: {}".format(group_idx)
                for param, pruned_axis, pruned_idx in group_idx:
                    pruned_num = int(round(len(pruned_idx) * ratio))
                    print pruned_num
                    pruned_params.append((
                        param, pruned_axis,
                        pruned_idx[:pruned_num]))  # [(name, axis, pruned_idx)]

        merge_pruned_params = {}
        for param, pruned_axis, pruned_idx in pruned_params:
            if param not in merge_pruned_params:
                merge_pruned_params[param] = {}
            if pruned_axis not in merge_pruned_params[param]:
                merge_pruned_params[param][pruned_axis] = []
            merge_pruned_params[param][pruned_axis].append(pruned_idx)

        for param_name in merge_pruned_params:
            for pruned_axis in merge_pruned_params[param_name]:
                pruned_idx = np.concatenate(merge_pruned_params[param_name][
                    pruned_axis])
                param = graph.var(param_name)
                if not lazy:
                    _logger.debug("{}\t{}\t{}".format(param.name(
                    ), pruned_axis, len(pruned_idx)))
                    if param_shape_backup is not None:
                        origin_shape = copy.deepcopy(param.shape())
                        param_shape_backup[param.name()] = origin_shape
                    new_shape = list(param.shape())
                    new_shape[pruned_axis] -= len(pruned_idx)
                    param.set_shape(new_shape)
                if not only_graph:
                    param_t = graph.var(param_name).data()
                    if param_backup is not None and (
                            param.name() not in param_backup):
                        param_backup[param.name()] = copy.deepcopy(
                            np.array(param_t))
                    try:
                        pruned_param = self._prune_tensor(
                            np.array(param_t),
                            pruned_idx,
                            pruned_axis=pruned_axis,
                            lazy=lazy)
                    except IndexError as e:
                        _logger.error("Pruning {}, but get [{}]".format(
                            param.name(), e))

                    graph.var(param_name).set_data(pruned_param, place=place)
        graph.update_groups_of_conv()
        graph.infer_shape()
        if isinstance(graph, DyGraph):
            return graph.module, param_backup, param_shape_backup
        else:
            return graph.program, param_backup, param_shape_backup

    def _cal_pruned_idx(self, graph, scope, param, ratio, axis):
        """
        Calculate the index to be pruned on axis by given pruning ratio.

        Args:
            name(str): The name of parameter to be pruned.
            param(np.array): The data of parameter to be pruned.
            ratio(float): The ratio to be pruned.
            axis(int): The axis to be used for pruning given parameter.
                       If it is None, the value in self.pruning_axis will be used.
                       default: None.

        Returns:
            list<int>: The indexes to be pruned on axis.
        """
        if self.criterion == 'l1_norm':
            param_t = np.array(scope.find_var(param).get_tensor())
            prune_num = int(round(param_t.shape[axis] * ratio))
            reduce_dims = [i for i in range(len(param_t.shape)) if i != axis]
            criterions = np.sum(np.abs(param_t), axis=tuple(reduce_dims))
            pruned_idx = criterions.argsort()[:prune_num]
        elif self.criterion == "batch_norm_scale":
            param_var = graph.var(param)
            conv_op = param_var.outputs()[0]
            conv_output = conv_op.outputs("Output")[0]
            bn_op = conv_output.outputs()[0]
            if bn_op is not None:
                bn_scale_param = bn_op.inputs("Scale")[0].name()
                bn_scale_np = np.array(
                    scope.find_var(bn_scale_param).get_tensor())
                prune_num = int(round(bn_scale_np.shape[axis] * ratio))
                pruned_idx = np.abs(bn_scale_np).argsort()[:prune_num]
            else:
                raise SystemExit(
                    "Can't find BatchNorm op after Conv op in Network.")
        return pruned_idx

    def _prune_tensor(self, tensor, pruned_idx, pruned_axis, lazy=False):
        """
        Pruning a array by indexes on given axis.

        Args:
            tensor(numpy.array): The target array to be pruned.
            pruned_idx(list<int>): The indexes to be pruned.
            pruned_axis(int): The axis of given array to be pruned on. 
            lazy(bool): True means setting the pruned elements to zero.
                        False means remove the pruned elements from memory.
                        default: False.

        Returns:
            numpy.array: The pruned array.
        """
        mask = np.zeros(tensor.shape[pruned_axis], dtype=bool)
        mask[pruned_idx] = True

        def func(data):
            return data[~mask]

        def lazy_func(data):
            data[mask] = 0
            return data

        if lazy:
            return np.apply_along_axis(lazy_func, pruned_axis, tensor)
        else:
            return np.apply_along_axis(func, pruned_axis, tensor)
