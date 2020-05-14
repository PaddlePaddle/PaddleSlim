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
from functools import reduce
import paddle.fluid as fluid
import copy
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

    def __init__(self,
                 criterion="l1_norm",
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
        graph = GraphWrapper(program.clone())
        param_backup = {} if param_backup else None
        param_shape_backup = {} if param_shape_backup else None

        visited = {}
        pruned_params = []
        for param, ratio in zip(params, ratios):
            if graph.var(param) is None:
                _logger.warn(
                    "Variable[{}] to be pruned is not in current graph.".
                    format(param))
                continue
            group = collect_convs([param], graph, visited)[0]  # [(name, axis)]
            if group is None or len(group) == 0:
                continue
            if only_graph and self.idx_selector.__name__ == "default_idx_selector":

                param_v = graph.var(param)
                pruned_num = int(round(param_v.shape()[0] * ratio))
                pruned_idx = [0] * pruned_num
                for name, axis in group:
                    pruned_params.append((name, axis, pruned_idx))

            else:
                assert ((not self.pruned_weights),
                        "The weights have been pruned once.")
                group_values = []
                for name, axis in group:
                    values = np.array(scope.find_var(name).get_tensor())
                    group_values.append((name, values, axis))

                scores = self.criterion(group_values,
                                        graph)  # [(name, axis, score)]

                pruned_params.extend(self.idx_selector(scores, ratio))

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
                    if param_shape_backup is not None:
                        origin_shape = copy.deepcopy(param.shape())
                        param_shape_backup[param.name()] = origin_shape
                    new_shape = list(param.shape())
                    new_shape[pruned_axis] -= len(pruned_idx)
                    param.set_shape(new_shape)
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
                    except IndexError as e:
                        _logger.error("Pruning {}, but get [{}]".format(
                            param.name(), e))

                    param_t.set(pruned_param, place)
        graph.update_groups_of_conv()
        graph.infer_shape()
        self.pruned_weights = (not only_graph)
        return graph.program, param_backup, param_shape_backup

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
