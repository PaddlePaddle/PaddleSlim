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

import os
import logging
import numpy as np
from ..core import Registry
from ..common import get_logger

__all__ = ["PRUNE_WORKER", "conv2d", "UnsupportOpError"]

_logger = get_logger(__name__, level=logging.INFO)

PRUNE_WORKER = Registry('prune_worker')

SKIPPED_OPS = ['shape', 'reduce_mean']

# operators in OPS_UNCHANGE_SHAPE will be visited by default worker
# who keep shape of output same with shape of input.
OPS_UNCHANGE_SHAPE = os.getenv('OPS_UNCHANGE_SHAPE', None)
OPS_UNCHANGE_SHAPE = [] if OPS_UNCHANGE_SHAPE is None else OPS_UNCHANGE_SHAPE.strip(
).split(",")
OPS_UNCHANGE_SHAPE += [
    'nearest_interp_v2',
    'roi_align',
    'sigmoid',
    'swish',
    'pad3d',
    'bilinear_interp_v2',
    'dropout',
    'cast',
    'hard_swish',
    'fused_softmax_mask_upper_triangle',
    'softmax',
    'hard_sigmoid',
]


class UnsupportOpError(Exception):
    pass


class PruneWorker(object):
    def __init__(self,
                 op,
                 pruned_params,
                 visited,
                 skip_stranger=True,
                 skip_vars=[]):
        """
        A wrapper of operator used to infer the information of all the related variables.

        Args:
            op(Operator): The operator to be pruned.
            pruned_params(list): The list to store the information of pruning that infered by worker.
            visited(dict): The auxiliary dict to record the visited operators and variables. The key is a encoded string of operator id and variable name.
            skip_stranger(bool): Whether to raise exception when visit unregistered operators that not in OPS_UNCHANGE_SHAPE. False means visit all unregistered operators by default waorker. Default: True.
            skip_vars(list<str>): The variables in 'skip_vars' and their relatives will be skipped. Default: [].

        Return: A instance of PruneWorker.
        """
        self.op = op
        self.pruned_params = pruned_params
        self.visited = visited
        self.skip_stranger = skip_stranger
        self.ops_unsupported = os.getenv('OPS_UNSUPPORTED', None)
        self.ops_unsupported = [] if self.ops_unsupported is None else self.ops_unsupported.strip(
        ).split(",")
        self.skip_vars = skip_vars

    def prune(self, var, pruned_axis, transforms):
        """ 
        Infer the shape of variables related with current operator, predecessor and successor. 
        It will search the graph to find all varibles related with `var` and record the information of pruning.
        Args:
            var(Variable): The root variable of searching. It can be the input or output of current operator.
            pruned_axis(int): The axis to be pruned of root variable.
            transforms(list<dict>): The transforms applied the the current variable/mask.
        """
        if var.name() in self.skip_vars:
            raise UnsupportOpError(
                "Variable {} was skipped.".format(var.name()))

        if self._visit(var, pruned_axis):
            self._prune(var, pruned_axis, transforms)

    def _visit(self, var, pruned_axis):
        key = "_".join([str(self.op.idx()), var.name()])
        key = "_".join([key, self.op.all_inputs()[0].name()])
        if pruned_axis not in self.visited:
            self.visited[pruned_axis] = {}
        if key in self.visited[pruned_axis]:
            return False
        else:
            self.visited[pruned_axis][key] = True
            return True

    def _visit_and_search(self, var, axis, transforms):
        self._visit(var, axis)
        if var.name() in self.skip_vars:
            raise UnsupportOpError(
                "Variable {} was skipped.".format(var.name()))
        pre_ops = var.inputs()
        for op in pre_ops:
            self._prune_op(op, var, axis, transforms)
        next_ops = var.outputs()
        for op in next_ops:
            self._prune_op(op, var, axis, transforms)

    def _prune(self, var, pruned_axis, transforms):
        raise NotImplementedError('Abstract method.')

    def _prune_op(self, op, var, pruned_axis, transforms, visited=None):
        if op.type().endswith("_grad"):
            return
        if visited is not None:
            self.visited = visited
        if op.type() in self.ops_unsupported:
            raise UnsupportOpError(
                "Unsupported operator named {}".format(op.type()))
        cls = PRUNE_WORKER.get(op.type())
        if cls is None:
            if op.type() in SKIPPED_OPS:
                return
            if op.type() in OPS_UNCHANGE_SHAPE or not self.skip_stranger:
                cls = PRUNE_WORKER.get("default_worker")
            else:
                raise UnsupportOpError(
                    "Unsupported operator named {}".format(op.type()))

        _logger.debug("\nfrom: {}\nto: {}\npruned_axis: {}; var: {}\ntrans: {}".
                      format(self.op, op, pruned_axis, var.name(), transforms))
        _logger.debug(
            f"visit {op.type()} by var [{var.name()}] on axis [{pruned_axis}];\t visited={self.visited}\n"
        )
        worker = cls(op, self.pruned_params, self.visited, self.skip_stranger)
        worker.skip_vars = self.skip_vars
        worker.prune(var, pruned_axis, transforms)

    def append_pruned_vars(self, var, axis, transforms):
        self.pruned_params.append((var, axis, transforms, self.op))


@PRUNE_WORKER.register
class reshape2(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(reshape2, self).__init__(op, pruned_params, visited,
                                       skip_stranger)

    def _valid_reshape2(self, shape):
        # case1: only reshape last several dimensions. e.g. [0,0,1,2] returns True while [1,0,0,1] returns False.
        changed = False
        for sh in shape:
            if sh == 0 and changed:
                return False
            if sh != 0:
                changed = True
        return True

    #NOTE: we might not need this assertion
    def _valid_pruned_axis(self, shape, pruned_axis):
        last_zero_index = -1
        for i in range(shape):
            if shape[i] == 0: last_zero_index = i
        if pruned_axis <= last_zero_index:
            return pruned_axis
        elif pruned_axis > last_zero_index:
            return pruned_axis

    def _get_idx_after_expanding_dims(self, pruned_axis, transforms,
                                      shorter_shape, longer_shape):
        assert len(shorter_shape) < len(
            longer_shape
        ), "length of {} must be smaller than length of {}.".format(
            shorter_shape, longer_shape)
        dim_old = shorter_shape[pruned_axis]
        dim_new = longer_shape[pruned_axis]
        k = dim_old / dim_new
        assert k == longer_shape[
            pruned_axis + 1], "k is {} while longer shape is {}[{}]".format(
                k, longer_shape, pruned_axis + 1)
        transforms = np.array(transforms, dtype='int32')
        pruned_rows = transforms / k
        pruned_cols = transforms % k
        new_transforms = []
        for row in range(dim_new):
            prune_this_row = row in pruned_rows
            prune_this_row = prune_this_row and (len(prued_rows) == k)
            if prune_this_row:
                new_transforms.append(row)

        return new_transforms

    def _get_idx_after_shrinking_dims(self, pruned_axis, transforms,
                                      longer_shape, shorter_shape):
        assert len(shorter_shape) < len(
            longer_shape
        ), "length of {} must be smaller than length of {}.".format(
            shorter_shape, longer_shape)

        dim_old = longer_shape[pruned_axis]
        dim_new = shorter_shape[pruned_axis]
        k = dim_new / dim_old
        assert k == longer_shape[pruned_axis + 1]

        new_transforms = []
        for row in range(dim_old):
            if row in transforms:
                new_transforms.expand(
                    [i for i in range(row * k, (row + 1) * k)])

        return new_transforms

    def _prune(self, var, pruned_axis, transforms):
        in_var = self.op.inputs("X")[0]
        out_var = self.op.outputs("Out")[0]
        xshape_var = self.op.outputs("XShape")[0]

        in_shape = in_var.shape()
        out_shape = out_var.shape()
        shape = self.op.attr("shape")
        assert self._valid_reshape2(
            shape), "we don't support the shape {} in pruning".format(shape)
        # assert self._valid_pruned_axis(shape, pruned_axis), "we don't support pruned axis is {} when shape is changing from {} to {}".format(pruned_axis, in_shape, out_shape)
        # self.append_pruned_vars(xshape_var, pruned_axis + 1, transforms)
        if var in self.op.inputs("X"):
            if (len(out_shape) > len(in_shape)):
                #self.op.set_attr('shape',
                #                 [0, 0, int(shape[2] * 0.875), shape[3]])
                transform = {"squeeze": out_shape[pruned_axis + 1]}
            elif (len(out_shape) < len(in_shape)):
                # self.op.set_attr('shape', [0, 0, int(shape[2] * 0.875)])
                transform = {"repeat": in_shape[pruned_axis + 1]}
            else:
                transform = {}
            self._visit_and_search(out_var, pruned_axis,
                                   transforms + [transform])
        elif var in self.op.outputs("Out"):
            if (len(in_shape) > len(out_shape)):
                # self.op.set_attr('shape', [0, 0, int(shape[2] * 0.875)])
                transform = {"squeeze": in_shape[pruned_axis + 1]}
            elif (len(in_shape) < len(in_shape)):
                #self.op.set_attr('shape',
                #                  [0, 0, int(shape[2] * 0.875), shape[3]])
                transform = {"repeat": out_shape[pruned_axis + 1]}
            elif len(in_shape) == 1 and len(
                    out_shape) == 4 and out_shape[pruned_axis] == in_shape[0]:
                transform = {}
                self.append_pruned_vars(in_var, 0, transforms)
            else:
                transform = {}
            self._visit_and_search(in_var, pruned_axis,
                                   transforms + [transform])


@PRUNE_WORKER.register
class transpose2(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(transpose2, self).__init__(op, pruned_params, visited,
                                         skip_stranger)

    def _prune(self, var, pruned_axis, transforms):
        axis = self.op.attr('axis')
        in_var = self.op.inputs("X")[0]
        out_var = self.op.outputs("Out")[0]
        if var in self.op.inputs("X"):
            new_pruned_axis = axis[pruned_axis]
            self._visit_and_search(out_var, new_pruned_axis, transforms)
        elif var in self.op.outputs("Out"):
            new_pruned_axis = axis[pruned_axis]
            self._visit_and_search(in_var, new_pruned_axis, transforms)


@PRUNE_WORKER.register
class conv2d(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(conv2d, self).__init__(op, pruned_params, visited, skip_stranger)

    def _is_depthwise_conv(self, op):
        data_format = self.op.attr("data_format")
        channel_axis = 1
        if data_format == "NHWC":
            channel_axis = 3

        filter_shape = self.op.inputs("Filter")[0].shape()
        input_shape = self.op.inputs("Input")[0].shape()
        num_channels = input_shape[channel_axis]
        groups = self.op.attr("groups")
        num_filters = filter_shape[0]
        return (num_channels == groups and num_channels != 1 and
                num_filters % num_channels == 0)

    def _prune(self, var, pruned_axis, transforms):
        if self._is_depthwise_conv(self.op):
            _logger.debug(f"Meet conv2d who is depthwise conv2d actually.")
            worker = depthwise_conv2d(
                self.op,
                self.pruned_params,
                visited=self.visited,
                skip_stranger=self.skip_stranger)
            return worker._prune(var, pruned_axis, transforms)

        data_format = self.op.attr("data_format")
        groups = self.op.attr("groups")
        channel_axis = 1
        if data_format == "NHWC":
            channel_axis = 3
        if var in self.op.inputs("Input"):
            assert pruned_axis == channel_axis, "The Input of conv2d can only be pruned at channel axis, but got {}; var: {}".format(
                pruned_axis, var.name())
            filter_var = self.op.inputs("Filter")[0]
            self.append_pruned_vars(filter_var, 1, transforms)
            if groups is None or groups == 1:
                self._visit_and_search(filter_var, 1, transforms)

        elif var in self.op.inputs("Filter"):
            assert pruned_axis in [0, 1]

            self.append_pruned_vars(var, pruned_axis, transforms)

            if groups is None or groups == 1 or pruned_axis == 0:
                self._visit_and_search(var, pruned_axis, transforms)

            if pruned_axis == 0:
                if len(self.op.inputs("Bias")) > 0:
                    self.append_pruned_vars(
                        self.op.inputs("Bias"), channel_axis, transforms)
                output_var = self.op.outputs("Output")[0]
                self._visit_and_search(output_var, channel_axis, transforms)

            elif pruned_axis == 1:
                input_var = self.op.inputs("Input")[0]
                self._visit_and_search(input_var, channel_axis, transforms)
        elif var in self.op.outputs("Output"):
            assert pruned_axis == channel_axis, "pruned_axis: {}; var: {}".format(
                pruned_axis, var.name())

            filter_var = self.op.inputs("Filter")[0]
            self._visit(filter_var, 0)
            self.append_pruned_vars(filter_var, 0, transforms)
            self._visit_and_search(filter_var, 0, transforms)
            if len(self.op.inputs("Bias")) > 0:
                self.append_pruned_vars(
                    self.op.inputs("Bias")[0], channel_axis, transforms)


@PRUNE_WORKER.register
class conv2d_transpose(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(conv2d_transpose, self).__init__(op, pruned_params, visited,
                                               skip_stranger)

    def _prune(self, var, pruned_axis, transforms):
        data_format = self.op.attr("data_format")
        channel_axis = 1
        if data_format == "NHWC":
            channel_axis = 3
        if var in self.op.inputs("Input"):
            assert pruned_axis == channel_axis, "The Input of conv2d can only be pruned at channel axis, but got {}; var: {}".format(
                pruned_axis, var.name())
            filter_var = self.op.inputs("Filter")[0]
            self._visit(filter_var, 0)
            self.append_pruned_vars(filter_var, 0, transforms)
            self._visit_and_search(filter_var, 0, transforms)

        elif var in self.op.inputs("Filter"):
            _logger.warn("Skip pruning output channels of conv2d_transpose!")
            return
        elif var in self.op.outputs("Output"):
            assert pruned_axis == channel_axis, "pruned_axis: {}; var: {}".format(
                pruned_axis, var.name())

            filter_var = self.op.inputs("Filter")[0]
            self._visit(filter_var, 1)
            self.append_pruned_vars(filter_var, 1, transforms)

            self._visit_and_search(filter_var, 1, transforms)

            if len(self.op.inputs("Bias")) > 0:
                self.append_pruned_vars(
                    self.op.inputs("Bias")[0], channel_axis, transforms)
            output_var = self.op.outputs("Output")[0]
            self._visit_and_search(output_var, channel_axis, transforms)


@PRUNE_WORKER.register
class batch_norm(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(batch_norm, self).__init__(op, pruned_params, visited,
                                         skip_stranger)

    def _prune(self, var, pruned_axis, transforms):
        if (var not in self.op.outputs("Y")) and (
                var not in self.op.inputs("X")):
            return

        if var in self.op.outputs("Y"):
            in_var = self.op.inputs("X")[0]
            self._visit_and_search(in_var, pruned_axis, transforms)

        for param in ["Scale", "Bias", "Mean", "Variance"]:
            param_var = self.op.inputs(param)[0]
            self._visit_and_search(param_var, 0, transforms)
            self.append_pruned_vars(param_var, 0, transforms)

        out_var = self.op.outputs("Y")[0]
        self._visit_and_search(out_var, pruned_axis, transforms)


@PRUNE_WORKER.register
class sync_batch_norm(batch_norm):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(sync_batch_norm, self).__init__(op, pruned_params, visited,
                                              skip_stranger)


class elementwise_op(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(elementwise_op, self).__init__(op, pruned_params, visited,
                                             skip_stranger)

    def _prune(self, var, pruned_axis, transforms):
        axis = self.op.attr("axis")
        if axis == -1:
            x = self.op.inputs("X")[0]
            y = self.op.inputs("Y")[0]
            axis = len(x.shape()) - len(y.shape())

        if var in self.op.outputs("Out"):
            for name in ["X", "Y"]:
                actual_axis = pruned_axis
                if name == "Y":
                    actual_axis = pruned_axis - axis
                in_var = self.op.inputs(name)[0]
                if len(in_var.shape()) == 1 and in_var.shape()[0] == 1:
                    continue

                # for bias
                if name == "Y" and actual_axis >= 0 and not (
                        len(in_var.shape()) == 1 and in_var.shape()[0] == 1):
                    self.append_pruned_vars(in_var, actual_axis, transforms)
                self._visit_and_search(in_var, actual_axis, transforms)

        else:
            if var in self.op.inputs("X"):
                in_var = self.op.inputs("Y")[0]
                y_pruned_axis = pruned_axis
                if len(in_var.shape()) != len(var.shape()):
                    assert (len(var.shape()) > len(in_var.shape()))
                    if axis == -1:
                        axis = len(var.shape()) - len(in_var.shape())
                    y_pruned_axis = pruned_axis - axis

                if y_pruned_axis >= 0 and not (len(in_var.shape()) == 1 and
                                               in_var.shape()[0] == 1):
                    self.append_pruned_vars(in_var, y_pruned_axis, transforms)
                    self._visit_and_search(in_var, y_pruned_axis, transforms)
            elif var in self.op.inputs("Y"):
                in_var = self.op.inputs("X")[0]
                if len(in_var.shape()) != len(var.shape()):
                    assert (len(var.shape()) < len(in_var.shape()))
                    pruned_axis = pruned_axis + axis
                if pruned_axis <= len(in_var.shape()):
                    self._visit_and_search(in_var, pruned_axis, transforms)

            out_var = self.op.outputs("Out")[0]
            self._visit_and_search(out_var, pruned_axis, transforms)


@PRUNE_WORKER.register
class elementwise_add(elementwise_op):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(elementwise_add, self).__init__(op, pruned_params, visited,
                                              skip_stranger)


@PRUNE_WORKER.register
class elementwise_sub(elementwise_op):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(elementwise_sub, self).__init__(op, pruned_params, visited,
                                              skip_stranger)


@PRUNE_WORKER.register
class elementwise_mul(elementwise_op):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(elementwise_mul, self).__init__(op, pruned_params, visited,
                                              skip_stranger)


@PRUNE_WORKER.register
class activation(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(activation, self).__init__(op, pruned_params, visited,
                                         skip_stranger)
        self.input_name = "X"
        self.output_name = "Out"

    def _prune(self, var, pruned_axis, transforms):
        if var in self.op.outputs(self.output_name):
            in_var = self.op.inputs(self.input_name)[0]
            self._visit_and_search(in_var, pruned_axis, transforms)
        if var in self.op.inputs(self.input_name):
            out_var = self.op.outputs(self.output_name)[0]
            self._visit_and_search(out_var, pruned_axis, transforms)


@PRUNE_WORKER.register
class default_worker(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(default_worker, self).__init__(op, pruned_params, visited,
                                             skip_stranger)

    def _prune(self, var, pruned_axis, transforms):
        if var in self.op.all_outputs():
            for in_var in self.op.all_inputs():
                if len(in_var.shape()) == len(var.shape()):
                    self._visit_and_search(in_var, pruned_axis, transforms)
        for out_var in self.op.all_outputs():
            if len(out_var.shape()) == len(var.shape()):
                self._visit_and_search(out_var, pruned_axis, transforms)


@PRUNE_WORKER.register
class bilinear_interp(activation):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(bilinear_interp, self).__init__(op, pruned_params, visited,
                                              skip_stranger)


@PRUNE_WORKER.register
class nearest_interp(activation):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(nearest_interp, self).__init__(op, pruned_params, visited,
                                             skip_stranger)


@PRUNE_WORKER.register
class relu(activation):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(relu, self).__init__(op, pruned_params, visited, skip_stranger)


@PRUNE_WORKER.register
class gelu(activation):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(gelu, self).__init__(op, pruned_params, visited, skip_stranger)


@PRUNE_WORKER.register
class leaky_relu(activation):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(leaky_relu, self).__init__(op, pruned_params, visited,
                                         skip_stranger)


@PRUNE_WORKER.register
class floor(activation):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(floor, self).__init__(op, pruned_params, visited, skip_stranger)


@PRUNE_WORKER.register
class relu6(activation):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(relu6, self).__init__(op, pruned_params, visited, skip_stranger)


@PRUNE_WORKER.register
class pool2d(activation):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(pool2d, self).__init__(op, pruned_params, visited, skip_stranger)


@PRUNE_WORKER.register
class sum(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(sum, self).__init__(op, pruned_params, visited, skip_stranger)

    def _prune(self, var, pruned_axis, transforms):
        if var in self.op.outputs("Out"):
            for in_var in self.op.inputs("X"):
                self._visit_and_search(in_var, pruned_axis, transforms)
        elif var in self.op.inputs("X"):
            for in_var in self.op.inputs("X"):
                if in_var != var:
                    self._visit_and_search(in_var, pruned_axis, transforms)
        out_var = self.op.outputs("Out")[0]
        self._visit_and_search(out_var, pruned_axis, transforms)


@PRUNE_WORKER.register
class split(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(split, self).__init__(op, pruned_params, visited, skip_stranger)
        self.in_var = op.inputs("X")[0]
        self.out_vars = op.outputs("Out")
        self.axis = op.attr("axis")
        self.num = op.attr("num")

    def _prune(self, var, pruned_axis, transforms):
        if var == self.in_var:
            if pruned_axis != self.axis:
                for i, out_var in enumerate(self.out_vars):
                    self._visit_and_search(out_var, pruned_axis, transforms)
            else:
                raise UnsupportOpError(
                    "Unsupport pruning input of split operator directly.")
        elif var in self.out_vars:
            if pruned_axis != self.axis:
                self._visit_and_search(self.in_var, pruned_axis, transforms)
            else:
                trans = {
                    "src_start": 0,
                    "src_end": var.shape()[pruned_axis],
                    "target_start": 0,
                    "target_end": self.in_var.shape()[pruned_axis],
                    "target_len": self.in_var.shape()[pruned_axis]
                }
                self._visit_and_search(self.in_var, pruned_axis,
                                       transforms + [trans])

            for out_var in self.out_vars:
                if var != out_var:
                    self._visit_and_search(out_var, pruned_axis, transforms)


@PRUNE_WORKER.register
class depthwise_conv2d(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(depthwise_conv2d, self).__init__(op, pruned_params, visited,
                                               skip_stranger)

    def _prune(self, var, pruned_axis, transforms):

        _filter = self.op.inputs("Filter")[0]
        _out = self.op.outputs("Output")[0]
        _in_var = self.op.inputs("Input")[0]
        _groups = self.op.attr("groups")
        data_format = self.op.attr("data_format")
        channel_axis = 1
        if data_format == "NHWC":
            channel_axis = 3
        if var == _in_var:
            assert pruned_axis == channel_axis, "The Input of conv2d can only be pruned at channel axis, but got {}".format(
                pruned_axis)
            # pruning number of filters
            assert (_filter.shape()[0] % _groups == 0)
            repeat = int(_filter.shape()[0] / _groups)
            self.append_pruned_vars(_filter, 0, transforms + [{
                "repeat": repeat
            }])
            # kernel_number * groups will be pruned by reducing groups
            self._visit_and_search(_filter, 0, transforms + [{
                "repeat": repeat
            }])
            # It will not pruning number of kernels in depthwise conv2d,
            # so it is not neccesary to search succeed operators.
            # self._visit_and_search(_filter, 1, transforms)
            self._visit(_filter, 1)
            self._visit_and_search(_out, channel_axis,
                                   transforms + [{
                                       "repeat": repeat
                                   }])
        elif var == _filter:
            assert pruned_axis == 0, "The filter of depthwise conv2d can only be pruned at axis 0."
            self.append_pruned_vars(_filter, 0, transforms)
            self._visit_and_search(_in_var, channel_axis, transforms)
            self._visit_and_search(_out, channel_axis, transforms)
        elif var == _out:
            assert pruned_axis == channel_axis, "The Input of conv2d can only be pruned at channel axis, but got {}".format(
                pruned_axis)
            self.append_pruned_vars(_filter, 0, transforms)
            self._visit_and_search(_filter, 0, transforms)
            # It will not pruning number of kernels in depthwise conv2d,
            # so it is not neccesary to search succeed operators.
            # self._visit_and_search(_filter, 1, transforms)
            self._visit(_filter, 1)
            self._visit_and_search(_in_var, channel_axis, transforms)


@PRUNE_WORKER.register
class mul(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(mul, self).__init__(op, pruned_params, visited, skip_stranger)

    def _prune(self, var, pruned_axis, trans):
        x_num_col_dims = self.op.attr("x_num_col_dims")
        y_num_col_dims = self.op.attr("y_num_col_dims")
        x = self.op.inputs("X")[0]
        y = self.op.inputs("Y")[0]
        out = self.op.outputs("Out")[0]
        x_shape = x.shape()
        y_shape = y.shape()
        if var == x:
            if y_num_col_dims > 1 and pruned_axis >= x_num_col_dims:
                raise UnsupportOpError(
                    "Unsupport pruning x of mul when y_num_col_dims > 1 and pruned_axis >= x_num_col_dims"
                )
            tile = 1
            repeat = 1
            if pruned_axis < x_num_col_dims:
                for i in range(0, pruned_axis):
                    tile *= x_shape[i]
                for i in range(pruned_axis + 1, x_num_col_dims):
                    repeat *= x_shape[i]
                self.append_pruned_vars(out, 0, trans + [{
                    "tile": tile,
                    "repeat": repeat
                }])
                self._visit_and_search(out, 0, trans + [{
                    "tile": tile,
                    "repeat": repeat
                }])
            else:
                for i in range(x_num_col_dims, pruned_axis):
                    tile *= x_shape[i]
                for i in range(pruned_axis + 1, len(x_shape)):
                    repeat *= x_shape[i]
                self.append_pruned_vars(y, 0, trans + [{
                    "tile": tile,
                    "repeat": repeat
                }])
                self._visit_and_search(y, 0, trans + [{
                    "tile": tile,
                    "repeat": repeat
                }])
        elif var == y:
            if (pruned_axis < y_num_col_dims) and (
                    1 < len(x_shape) - x_num_col_dims
            ) and max(x_shape[x_num_col_dims:]) != np.prod(
                    y_shape[:y_num_col_dims]):
                raise UnsupportOpError(
                    "Unsupport pruning y of mul when pruned_axis < y_num_col_dims and 1 < len(x_shape) - x_num_col_dims."
                )

            tile = 1
            repeat = 1

            self.append_pruned_vars(var, pruned_axis, trans)
            self._visit_and_search(var, pruned_axis, trans)

            if pruned_axis >= y_num_col_dims:
                for i in range(y_num_col_dims, pruned_axis):
                    tile *= y_shape[i]
                for i in range(pruned_axis + 1, len(y_shape)):
                    repeat *= y_shape[i]
                self.append_pruned_vars(out, 1, trans + [{
                    "tile": tile,
                    "repeat": repeat
                }])
                self._visit_and_search(out, 1, trans + [{
                    "tile": tile,
                    "repeat": repeat
                }])
            else:
                for i in range(0, pruned_axis):
                    tile *= y_shape[i]
                for i in range(pruned_axis + 1, y_num_col_dims):
                    repeat *= y_shape[i]
                new_pruned_axis = int(
                    np.argmax(x_shape[x_num_col_dims:])) + x_num_col_dims
                self.append_pruned_vars(
                    x,
                    # len(x_shape) - 1, trans + [{
                    new_pruned_axis,
                    trans + [{
                        "tile": tile,
                        "repeat": repeat
                    }])
                self._visit_and_search(
                    x,
                    # len(x_shape) - 1, trans + [{
                    new_pruned_axis,
                    trans + [{
                        "tile": tile,
                        "repeat": repeat
                    }])
        elif var == out:
            if (pruned_axis == 0 and x_num_col_dims != 1) or (
                    pruned_axis == 1 and (len(y_shape) - y_num_col_dims) != 1):
                raise UnsupportOpError(
                    "Unsupport pruning out of mul when pruned_axis={}; x_num_col_dims: {}; y_num_col_dims: {}; y_shape: {}.".
                    format(pruned_axis, x_num_col_dims, y_num_col_dims,
                           y_shape))

            if pruned_axis == 0:
                self.append_pruned_vars(x, 0, trans)
                self._visit_and_search(x, 0, trans)
            elif pruned_axis == 1:
                self.append_pruned_vars(y, len(y_shape) - 1, trans)
                self._visit_and_search(y, len(y_shape) - 1, trans)


@PRUNE_WORKER.register
class matmul(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(matmul, self).__init__(op, pruned_params, visited, skip_stranger)

    def _prune(self, var, pruned_axis, transforms):
        x = self.op.inputs("X")[0]
        y = self.op.inputs("Y")[0]
        out = self.op.outputs("Out")[0]
        x_shape_len = len(x.shape())
        y_shape_len = len(y.shape())
        mappings = []
        if x_shape_len == 1 and y_shape_len == 1:
            mappings = [(0, 0, 0)]
        elif x_shape_len == 1 and y_shape_len == 2:
            mappings = [(0, 0, -1), (-1, 1, 0)]
        elif x_shape_len == 2 and y_shape_len == 2:
            mappings = [(0, -1, 0), (1, 0, -1), (-1, 1, 1)]
        elif x_shape_len == 3 and y_shape_len == 1:
            mappings = [(1, -1, 1), (2, 0, -1)]
        elif x_shape_len == 2 and y_shape_len == 3:
            mappings = [(0, -1, 1), (1, 1, -1), (-1, 2, 2)]
        elif x_shape_len == 3 and y_shape_len == 2:
            mappings = [(2, 0, -1), (-1, 1, 2)]
        elif x_shape_len == 4 and y_shape_len == 4:
            mappings = [(1, 1, 1)]
        elif x_shape_len >= 3 and y_shape_len >= 3:
            mappings = [(x_shape_len - 2, -1, x_shape_len - 2),
                        (x_shape_len - 1, x_shape_len - 2,
                         -1), (-1, x_shape_len - 1, x_shape_len - 1)]
        if var == x:
            for x_i, y_i, out_i in mappings:
                if pruned_axis == x_i:
                    if y_i != -1:
                        self.append_pruned_vars(y, y_i, transforms)
                        self._visit_and_search(y, y_i, transforms)
                    if out_i != -1:
                        self._visit_and_search(out, out_i, transforms)
                    break
        if var == y:
            for x_i, y_i, out_i in mappings:
                if pruned_axis == y_i:
                    if x_i != -1:
                        self.append_pruned_vars(x, x_i, transforms)
                        self._visit_and_search(x, x_i, transforms)
                    if 'w_' in var.name():
                        self.append_pruned_vars(var, y_i, transforms)
                        self._visit_and_search(var, y_i, transforms)
                    if out_i != -1:
                        self._visit_and_search(out, out_i, transforms)
                    break
        if var == out:
            for x_i, y_i, out_i in mappings:
                if pruned_axis == out_i:
                    if x_i != -1:
                        self.append_pruned_vars(x, x_i, transforms)
                        self._visit_and_search(x, x_i, transforms)
                    if y_i != -1:
                        self.append_pruned_vars(y, y_i, transforms)
                        self._visit_and_search(y, y_i, transforms)
                    break


@PRUNE_WORKER.register
class matmul_v2(matmul):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(matmul_v2, self).__init__(op, pruned_params, visited,
                                        skip_stranger)


@PRUNE_WORKER.register
class scale(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(scale, self).__init__(op, pruned_params, visited, skip_stranger)

    def _prune(self, var, pruned_axis, transforms):
        if var in self.op.inputs("X"):
            out_var = self.op.outputs("Out")[0]
            self._visit_and_search(out_var, pruned_axis, transforms)
        elif var in self.op.outputs("Out"):
            in_var = self.op.inputs("X")[0]
            self._visit_and_search(in_var, pruned_axis, transforms)


@PRUNE_WORKER.register
class momentum(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(momentum, self).__init__(op, pruned_params, visited,
                                       skip_stranger)

    def _prune(self, var, pruned_axis, transforms):
        if var in self.op.inputs("Param"):
            velocity_var = self.op.inputs("Velocity")[0]
            self.append_pruned_vars(velocity_var, pruned_axis, transforms)


@PRUNE_WORKER.register
class adam(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(adam, self).__init__(op, pruned_params, visited, skip_stranger)

    def _prune(self, var, pruned_axis, transforms):
        if var in self.op.inputs("Param"):
            moment1_var = self.op.inputs("Moment1")[0]
            self.append_pruned_vars(moment1_var, pruned_axis, transforms)
            moment2_var = self.op.inputs("Moment2")[0]
            self.append_pruned_vars(moment2_var, pruned_axis, transforms)


@PRUNE_WORKER.register
class affine_channel(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(affine_channel, self).__init__(op, pruned_params, visited,
                                             skip_stranger)

    def _prune(self, var, pruned_axis, transforms):
        if (var not in self.op.outputs("Out")) and (
                var not in self.op.inputs("X")):
            return

        if var in self.op.outputs("Out"):
            in_var = self.op.inputs("X")[0]
            self._visit_and_search(in_var, pruned_axis, transforms)

        for param in ["Scale", "Bias"]:
            param_var = self.op.inputs(param)[0]
            self._visit_and_search(param_var, 0, transforms)
            self.append_pruned_vars(param_var, 0, transforms)

        out_var = self.op.outputs("Out")[0]
        self._visit_and_search(out_var, pruned_axis, transforms)


@PRUNE_WORKER.register
class flatten_contiguous_range(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(flatten_contiguous_range, self).__init__(op, pruned_params,
                                                       visited, skip_stranger)

    def _prune(self, var, pruned_axis, transforms):
        start_axis = self.op.attr("start_axis")
        stop_axis = self.op.attr("stop_axis")
        if var in self.op.outputs("Out"):
            out_var = self.op.outputs("Out")[0]
            in_var = self.op.inputs("X")[0]
            assert pruned_axis == start_axis and pruned_axis == int(
                np.argmax(in_var.shape()[1:]) + 1)
            self._visit_and_search(in_var, pruned_axis, transforms)

        if var in self.op.inputs("X"):
            out_var = self.op.outputs("Out")[0]
            in_var = self.op.inputs("X")[0]
            stride = 1
            out_pruned_axis = pruned_axis
            if pruned_axis >= start_axis and pruned_axis <= stop_axis:
                out_pruned_axis = start_axis
                for i in range(start_axis, stop_axis + 1):
                    if i != pruned_axis:
                        stride *= in_var.shape()[i]
            elif pruned_axis > stop_axis:
                out_pruned_axis = start_axis + pruned_axis - stop_axis

            self._visit(in_var, pruned_axis)
            transform = {'stride': stride}
            self._visit_and_search(out_var, out_pruned_axis,
                                   transforms + [transform])


@PRUNE_WORKER.register
class squeeze2(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(squeeze2, self).__init__(op, pruned_params, visited,
                                       skip_stranger)

    def _prune(self, var, pruned_axis, transforms):

        axes = self.op.attr("axes")
        in_var = self.op.inputs("X")[0]
        out_var = self.op.outputs("Out")[0]
        if axes is None or len(axes) == 0:
            axes = [i for i, axis in enumerate(in_var.shape()) if axis == 1]
        squeeze_num = 0
        if in_var == var:
            for axis in axes:
                assert axis != pruned_axis, "Can not pruning axis that will be squeezed."
                if axis < pruned_axis:
                    squeeze_num += 1
            pruned_axis -= squeeze_num
            self._visit_and_search(out_var, pruned_axis, transforms)
        elif out_var == var:
            for axis in axes:
                if axis <= pruned_axis:
                    pruned_axis += 1
            self._visit_and_search(in_var, pruned_axis, transforms)


@PRUNE_WORKER.register
class unsqueeze2(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(unsqueeze2, self).__init__(op, pruned_params, visited,
                                         skip_stranger)

    def _prune(self, var, pruned_axis, transforms):

        axes = self.op.attr("axes")
        in_var = self.op.inputs("X")[0]
        out_var = self.op.outputs("Out")[0]
        assert (axes is not None)

        squeeze_num = 0
        if in_var == var:
            for axis in axes:
                if axis <= pruned_axis:
                    pruned_axis += 1
            self._visit_and_search(out_var, pruned_axis, transforms)
        elif out_var == var:
            for axis in axes:
                if axis < pruned_axis:
                    squeeze_num += 1
            pruned_axis -= squeeze_num
            self._visit_and_search(in_var, pruned_axis, transforms)


@PRUNE_WORKER.register
class average_accumulates(PruneWorker):
    def __init__(self, op, pruned_params, visited, skip_stranger):
        super(average_accumulates, self).__init__(op, pruned_params, visited,
                                                  skip_stranger)

    def _prune(self, var, pruned_axis, transforms):
        in_var = self.op.inputs("param")[0]
        out_var_1 = self.op.outputs("out_sum_1")[0]
        out_var_2 = self.op.outputs("out_sum_2")[0]
        out_var_3 = self.op.outputs("out_sum_3")[0]
        if in_var == var:
            self.append_pruned_vars(out_var_1, pruned_axis, transforms)
            self.append_pruned_vars(out_var_2, pruned_axis, transforms)
            self.append_pruned_vars(out_var_3, pruned_axis, transforms)
