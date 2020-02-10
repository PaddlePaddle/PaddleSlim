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
import numpy as np
import paddle.fluid as fluid
import copy
from ..core import VarWrapper, OpWrapper, GraphWrapper
from .prune_walker import conv2d as conv2d_walker
from ..common import get_logger

__all__ = ["Pruner"]

_logger = get_logger(__name__, level=logging.INFO)


class Pruner():
    """The pruner used to prune channels of convolution.

    Args:
        criterion(str): the criterion used to sort channels for pruning. It only supports 'l1_norm' currently.

    """

    def __init__(self, criterion="l1_norm"):
        self.criterion = criterion

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
            if only_graph:
                param_v = graph.var(param)
                pruned_num = int(round(param_v.shape()[0] * ratio))
                pruned_idx = [0] * pruned_num
            else:
                param_t = np.array(scope.find_var(param).get_tensor())
                pruned_idx = self._cal_pruned_idx(param_t, ratio, axis=0)
            param = graph.var(param)
            conv_op = param.outputs()[0]
            walker = conv2d_walker(
                conv_op, pruned_params=pruned_params, visited=visited)
            walker.prune(param, pruned_axis=0, pruned_idx=pruned_idx)

        merge_pruned_params = {}
        for param, pruned_axis, pruned_idx in pruned_params:
            if param.name() not in merge_pruned_params:
                merge_pruned_params[param.name()] = {}
            if pruned_axis not in merge_pruned_params[param.name()]:
                merge_pruned_params[param.name()][pruned_axis] = []
            merge_pruned_params[param.name()][pruned_axis].append(pruned_idx)

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
        return graph.program, param_backup, param_shape_backup

    def _cal_pruned_idx(self, param, ratio, axis):
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
        prune_num = int(round(param.shape[axis] * ratio))
        reduce_dims = [i for i in range(len(param.shape)) if i != axis]
        if self.criterion == 'l1_norm':
            criterions = np.sum(np.abs(param), axis=tuple(reduce_dims))
        pruned_idx = criterions.argsort()[:prune_num]
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
