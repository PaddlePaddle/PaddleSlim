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
import copy
import numpy as np
from functools import reduce
from ..core import VarWrapper, OpWrapper, GraphWrapper
from .group_param import collect_convs
from .criterion import CRITERION
from .idx_selector import IDX_SELECTOR
from ..common import get_logger

__all__ = ["Pruner"]

_logger = get_logger(__name__, level=logging.INFO)


class Pruner():
    """The pruner used to prune channels of convolution.

    Args:
        criterion(str|function): the criterion used to sort channels for pruning.
        idx_selector(str|function): 

    """

    def __init__(self, criterion="l1_norm",
                 idx_selector="default_idx_selector"):
        if isinstance(criterion, str):
            self.criterion = CRITERION.get(criterion)
        else:
            self.criterion = criterion
        if isinstance(idx_selector, str):
            self.idx_selector = IDX_SELECTOR.get(idx_selector)
        else:
            self.idx_selector = idx_selector

        self.pruned_weights = False

    def prune(self,
              program,
              scope,
              params,
              ratios,
              place=None,
              lazy=False,
              only_graph=False,
              param_backup=False,
              param_shape_backup=False):
        """Pruning the given parameters.

        Args:

            program(paddle.static.Program): The program to be pruned.
            scope(paddle.static.Scope): The scope storing paramaters to be pruned.
            params(list<str>): A list of parameter names to be pruned.
            ratios(list<float>): A list of ratios to be used to pruning parameters.
            place(paddle.CUDAPlace||paddle.CPUPlace): The device place of filter parameters. Defalut: None.
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
        graph = GraphWrapper(program.clone())
        param_backup = {} if param_backup else None
        param_shape_backup = {} if param_shape_backup else None

        pruned_params = []
        visited = {}
        for param, ratio in zip(params, ratios):
            _logger.info("pruning: {}".format(param))
            if graph.var(param) is None:
                _logger.warn(
                    "Variable[{}] to be pruned is not in current graph.".format(
                        param))
                continue
            group = collect_convs([param], graph,
                                  visited)[0]  # [(name, axis, pruned_idx)]
            if group is None or len(group) == 0:
                continue
            assert (
                not self.pruned_weights), "The weights have been pruned once."
            group_values = []
            for name, axis, pruned_idx in group:
                var = scope.find_var(name)
                if var is not None:
                    values = np.array(var.get_tensor())
                    group_values.append((name, values, axis, pruned_idx))

            scores = self.criterion(group_values,
                                    graph)  # [(name, axis, score, pruned_idx)]
            g = self._transform(self.idx_selector(scores, ratio))
            pruned_params.extend(g)

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
                    _logger.debug("{}\t{}\t{}\t{}".format(
                        param.name(), pruned_axis,
                        param.shape()[pruned_axis], len(pruned_idx)))
                    origin_shape = copy.deepcopy(param.shape())
                    if param_shape_backup is not None:
                        param_shape_backup[param.name()] = origin_shape
                    new_shape = list(param.shape())
                    new_shape[pruned_axis] -= len(pruned_idx)
                    param.set_shape(new_shape)
                    # update groups of depthwise conv2d
                    for op in param.outputs():
                        if op.type() in ["conv2d", "depthwise_conv2d"
                                         ] and op.attr("groups") > 1:
                            assert origin_shape[
                                1] == 1, "Only support for depthwise when groups > 1."
                            new_groups = int(
                                op.attr("groups") * new_shape[pruned_axis] /
                                origin_shape[pruned_axis])
                            _logger.debug(
                                "change groups of conv({}) from {} to {}; origin_shape: {}; new_shape: {}".format(param.name(), op.attr('groups'), new_groups, origin_shape, new_shape)
                            )
                            op.set_attr("groups", new_groups)

                if not only_graph:
                    param_t = scope.find_var(param.name()).get_tensor()
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
                        param_t.set(pruned_param, place)
                    except IndexError as e:
                        _logger.error("Pruning {}, but get [{}]".format(
                            param.name(), e))

        graph.infer_shape()
        self.pruned_weights = (not only_graph)
        return graph.program, param_backup, param_shape_backup

    def _transform(self, group):
        ret = []
        for name, axis, pruned_idx, transforms in group:
            src = pruned_idx
            for trans in transforms:
                src_start = trans['src_start']
                src_end = trans['src_end']
                target_start = trans['target_start']
                target_end = trans['target_end']
                target = []
                for idx in src:
                    if idx >= src_start and idx < src_end:
                        idx -= src_start
                        idx += target_start
                        if idx < target_end:
                            target.append(idx)
                src = target
            ret.append((name, axis, src))
        return ret

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
