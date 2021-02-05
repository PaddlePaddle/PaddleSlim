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
from ..core import Registry
from ..common import get_logger

__all__ = ["PRUNE_WORKER", "conv2d"]

_logger = get_logger(__name__, level=logging.INFO)

PRUNE_WORKER = Registry('prune_worker')

SKIP_OPS = ["conditional_block"]


class PruneWorker(object):
    def __init__(self, op, pruned_params=[], visited={}):
        """
        A wrapper of operator used to infer the information of all the related variables.

        Args:
            op(Operator): The operator to be pruned.
            pruned_params(list): The list to store the information of pruning that infered by walker.
            visited(dict): The auxiliary dict to record the visited operators and variables. The key is a encoded string of operator id and variable name.

        Return: A instance of PruneWalker.
        """
        self.op = op
        self.pruned_params = pruned_params
        self.visited = visited

    def prune(self, var, pruned_axis, pruned_idx):
        """ 
        Infer the shape of variables related with current operator, predecessor and successor. 
        It will search the graph to find all varibles related with `var` and record the information of pruning.
        Args:
            var(Variable): The root variable of searching. It can be the input or output of current operator.
            pruned_axis(int): The axis to be pruned of root variable.
            pruned_idx(int): The indexes to be pruned in `pruned_axis` of root variable.
        """
        if self._visit(var, pruned_axis):
            self._prune(var, pruned_axis, pruned_idx)

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
        pre_ops = var.inputs()
        for op in pre_ops:
            self._prune_op(op, var, axis, transforms)
        next_ops = var.outputs()
        for op in next_ops:
            self._prune_op(op, var, axis, transforms)

    def _prune(self, var, pruned_axis, pruned_idx):
        raise NotImplementedError('Abstract method.')

    def _prune_op(self, op, var, pruned_axis, pruned_idx, visited=None):
        if op.type().endswith("_grad"):
            return
        if visited is not None:
            self.visited = visited
        cls = PRUNE_WORKER.get(op.type())
        if cls is None:
            if op.type() in SKIP_OPS:
                _logger.warn("Skip operator [{}]".format(op.type()))
                return

#            _logger.warn(
#                "{} op will be pruned by default walker to keep the shapes of input and output being same because its walker is not registered.".
#                format(op.type()))
            cls = PRUNE_WORKER.get("default_walker")
        _logger.debug("\nfrom: {}\nto: {}\npruned_axis: {}; var: {}".format(
            self.op, op, pruned_axis, var.name()))
        _logger.debug(
            "visit {} by var [{}] on axis [{}];\t visited={}\n".format(op.type(), var.name(), pruned_axis, self.visited)
        )
        walker = cls(op, pruned_params=self.pruned_params, visited=self.visited)
        walker.prune(var, pruned_axis, pruned_idx)


@PRUNE_WORKER.register
class conv2d(PruneWorker):
    def __init__(self, op, pruned_params, visited={}):
        super(conv2d, self).__init__(op, pruned_params, visited)

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

    def _prune(self, var, pruned_axis, pruned_idx):

        if self._is_depthwise_conv(self.op):
            _logger.debug("Meet conv2d who is depthwise conv2d actually.")
            walker = depthwise_conv2d(
                self.op, self.pruned_params, visited=self.visited)
            walker._prune(var, pruned_axis, pruned_idx)
            return

        data_format = self.op.attr("data_format")
        channel_axis = 1
        if data_format == "NHWC":
            channel_axis = 3
        if var in self.op.inputs("Input"):
            assert pruned_axis == channel_axis, "The Input of conv2d can only be pruned at channel axis, but got {}; var: {}".format(
                pruned_axis, var.name())
            filter_var = self.op.inputs("Filter")[0]
            self._visit(filter_var, 1)
            self.pruned_params.append((filter_var, 1, pruned_idx))
            for op in filter_var.outputs():
                self._prune_op(op, filter_var, 1, pruned_idx)

        elif var in self.op.inputs("Filter"):
            assert pruned_axis in [0, 1]

            self.pruned_params.append((var, pruned_axis, pruned_idx))

            for op in var.outputs():
                self._prune_op(op, var, pruned_axis, pruned_idx)

            if pruned_axis == 0:
                if len(self.op.inputs("Bias")) > 0:
                    self.pruned_params.append(
                        (self.op.inputs("Bias"), channel_axis, pruned_idx))
                output_var = self.op.outputs("Output")[0]
                self._visit(output_var, channel_axis)
                next_ops = output_var.outputs()
                for op in next_ops:
                    self._prune_op(op, output_var, channel_axis, pruned_idx)

            elif pruned_axis == 1:
                input_var = self.op.inputs("Input")[0]
                self._visit(input_var, channel_axis)
                pre_ops = input_var.inputs()
                for op in pre_ops:
                    self._prune_op(op, input_var, channel_axis, pruned_idx)
        elif var in self.op.outputs("Output"):
            assert pruned_axis == channel_axis, "pruned_axis: {}; var: {}".format(
                pruned_axis, var.name())

            filter_var = self.op.inputs("Filter")[0]
            self._visit(filter_var, 0)

            self.pruned_params.append((filter_var, 0, pruned_idx))

            for op in filter_var.outputs():
                self._prune_op(op, filter_var, 0, pruned_idx)

            if len(self.op.inputs("Bias")) > 0:
                self.pruned_params.append(
                    (self.op.inputs("Bias")[0], channel_axis, pruned_idx))


@PRUNE_WORKER.register
class conv2d_transpose(PruneWorker):
    def __init__(self, op, pruned_params, visited={}):
        super(conv2d_transpose, self).__init__(op, pruned_params, visited)

    def _prune(self, var, pruned_axis, pruned_idx):
        data_format = self.op.attr("data_format")
        channel_axis = 1
        if data_format == "NHWC":
            channel_axis = 3
        if var in self.op.inputs("Input"):
            assert pruned_axis == channel_axis, "The Input of conv2d can only be pruned at channel axis, but got {}; var: {}".format(
                pruned_axis, var.name())
            filter_var = self.op.inputs("Filter")[0]
            self._visit(filter_var, 0)
            self.pruned_params.append((filter_var, 0, pruned_idx))
            for op in filter_var.outputs():
                self._prune_op(op, filter_var, 0, pruned_idx)

        elif var in self.op.inputs("Filter"):
            _logger.warn("Skip pruning output channels of conv2d_transpose!")
            return
        elif var in self.op.outputs("Output"):
            assert pruned_axis == channel_axis, "pruned_axis: {}; var: {}".format(
                pruned_axis, var.name())

            filter_var = self.op.inputs("Filter")[0]
            self._visit(filter_var, 1)

            self.pruned_params.append((filter_var, 1, pruned_idx))

            for op in filter_var.outputs():
                self._prune_op(op, filter_var, 1, pruned_idx)

            if len(self.op.inputs("Bias")) > 0:
                self.pruned_params.append(
                    (self.op.inputs("Bias")[0], channel_axis, pruned_idx))

            output_var = self.op.outputs("Output")[0]
            next_ops = output_var.outputs()
            for op in next_ops:
                self._prune_op(op, output_var, channel_axis, pruned_idx)


@PRUNE_WORKER.register
class batch_norm(PruneWorker):
    def __init__(self, op, pruned_params, visited):
        super(batch_norm, self).__init__(op, pruned_params, visited)

    def _prune(self, var, pruned_axis, pruned_idx):
        if (var not in self.op.outputs("Y")) and (
                var not in self.op.inputs("X")):
            return

        if var in self.op.outputs("Y"):
            in_var = self.op.inputs("X")[0]
            self._visit(in_var, pruned_axis)
            pre_ops = in_var.inputs()
            for op in pre_ops:
                self._prune_op(op, in_var, pruned_axis, pruned_idx)

        for param in ["Scale", "Bias", "Mean", "Variance"]:
            param_var = self.op.inputs(param)[0]
            for op in param_var.outputs():
                self._prune_op(op, param_var, 0, pruned_idx)
            self.pruned_params.append((param_var, 0, pruned_idx))

        out_var = self.op.outputs("Y")[0]
        self._visit(out_var, pruned_axis)
        next_ops = out_var.outputs()
        for op in next_ops:
            self._prune_op(op, out_var, pruned_axis, pruned_idx)


@PRUNE_WORKER.register
class sync_batch_norm(batch_norm):
    def __init__(self, op, pruned_params, visited):
        super(sync_batch_norm, self).__init__(op, pruned_params, visited)


class elementwise_op(PruneWorker):
    def __init__(self, op, pruned_params, visited):
        super(elementwise_op, self).__init__(op, pruned_params, visited)

    def _prune(self, var, pruned_axis, pruned_idx):
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
                    self.pruned_params.append((in_var, actual_axis, pruned_idx))
                self._visit_and_search(in_var, actual_axis, pruned_idx)

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
                    self.pruned_params.append(
                        (in_var, y_pruned_axis, pruned_idx))
                    self._visit_and_search(in_var, y_pruned_axis, pruned_idx)
            elif var in self.op.inputs("Y"):
                in_var = self.op.inputs("X")[0]
                if len(in_var.shape()) != len(var.shape()):
                    assert (len(var.shape()) < len(in_var.shape()))
                    pruned_axis = pruned_axis + axis
                if pruned_axis <= len(in_var.shape()):
                    self._visit_and_search(in_var, pruned_axis, pruned_idx)

            out_var = self.op.outputs("Out")[0]
            self._visit_and_search(out_var, pruned_axis, pruned_idx)


@PRUNE_WORKER.register
class elementwise_add(elementwise_op):
    def __init__(self, op, pruned_params, visited):
        super(elementwise_add, self).__init__(op, pruned_params, visited)


@PRUNE_WORKER.register
class elementwise_sub(elementwise_op):
    def __init__(self, op, pruned_params, visited):
        super(elementwise_sub, self).__init__(op, pruned_params, visited)


@PRUNE_WORKER.register
class elementwise_mul(elementwise_op):
    def __init__(self, op, pruned_params, visited):
        super(elementwise_mul, self).__init__(op, pruned_params, visited)


@PRUNE_WORKER.register
class activation(PruneWorker):
    def __init__(self, op, pruned_params, visited):
        super(activation, self).__init__(op, pruned_params, visited)
        self.input_name = "X"
        self.output_name = "Out"

    def _prune(self, var, pruned_axis, pruned_idx):
        if var in self.op.outputs(self.output_name):
            in_var = self.op.inputs(self.input_name)[0]
            self._visit_and_search(in_var, pruned_axis, pruned_idx)
        if var in self.op.inputs(self.input_name):
            out_var = self.op.outputs(self.output_name)[0]
            self._visit_and_search(out_var, pruned_axis, pruned_idx)


@PRUNE_WORKER.register
class default_walker(PruneWorker):
    def __init__(self, op, pruned_params, visited):
        super(default_walker, self).__init__(op, pruned_params, visited)

    def _prune(self, var, pruned_axis, pruned_idx):
        if var in self.op.all_outputs():
            for in_var in self.op.all_inputs():
                if len(in_var.shape()) == len(var.shape()):
                    self._visit_and_search(in_var, pruned_axis, pruned_idx)
        for out_var in self.op.all_outputs():
            if len(out_var.shape()) == len(var.shape()):
                self._visit_and_search(out_var, pruned_axis, pruned_idx)


@PRUNE_WORKER.register
class uniform_random_batch_size_like(activation):
    def __init__(self, op, pruned_params, visited):
        super(uniform_random_batch_size_like, self).__init__(op, pruned_params,
                                                             visited)
        self.input_name = "Input"
        self.output_name = "Out"


@PRUNE_WORKER.register
class bilinear_interp(activation):
    def __init__(self, op, pruned_params, visited):
        super(bilinear_interp, self).__init__(op, pruned_params, visited)


@PRUNE_WORKER.register
class nearest_interp(activation):
    def __init__(self, op, pruned_params, visited):
        super(nearest_interp, self).__init__(op, pruned_params, visited)


@PRUNE_WORKER.register
class relu(activation):
    def __init__(self, op, pruned_params, visited):
        super(relu, self).__init__(op, pruned_params, visited)


@PRUNE_WORKER.register
class leaky_relu(activation):
    def __init__(self, op, pruned_params, visited):
        super(leaky_relu, self).__init__(op, pruned_params, visited)


@PRUNE_WORKER.register
class floor(activation):
    def __init__(self, op, pruned_params, visited):
        super(floor, self).__init__(op, pruned_params, visited)


@PRUNE_WORKER.register
class relu6(activation):
    def __init__(self, op, pruned_params, visited):
        super(relu6, self).__init__(op, pruned_params, visited)


@PRUNE_WORKER.register
class pool2d(activation):
    def __init__(self, op, pruned_params, visited):
        super(pool2d, self).__init__(op, pruned_params, visited)


@PRUNE_WORKER.register
class sum(PruneWorker):
    def __init__(self, op, pruned_params, visited):
        super(sum, self).__init__(op, pruned_params, visited)

    def _prune(self, var, pruned_axis, pruned_idx):
        if var in self.op.outputs("Out"):
            for in_var in self.op.inputs("X"):
                pre_ops = in_var.inputs()
                for op in pre_ops:
                    self._prune_op(op, in_var, pruned_axis, pruned_idx)
        elif var in self.op.inputs("X"):
            for in_var in self.op.inputs("X"):
                if in_var != var:
                    pre_ops = in_var.inputs()
                    for op in pre_ops:
                        self._prune_op(op, in_var, pruned_axis, pruned_idx)
        out_var = self.op.outputs("Out")[0]
        self._visit(out_var, pruned_axis)
        next_ops = out_var.outputs()
        for op in next_ops:
            self._prune_op(op, out_var, pruned_axis, pruned_idx)


@PRUNE_WORKER.register
class concat(PruneWorker):
    def __init__(self, op, pruned_params, visited):
        super(concat, self).__init__(op, pruned_params, visited)

    def _prune(self, var, pruned_axis, transforms):
        axis = self.op.attr("axis")
        if var in self.op.outputs("Out"):
            self._visit(var, pruned_axis)
            start = 0
            if axis == pruned_axis:
                for _, in_var in enumerate(self.op.inputs("X")):
                    idx = []
                    transoform = {
                        'src_start': start,
                        'src_end': start + in_var.shape()[pruned_axis],
                        'target_start': 0,
                        'target_end': in_var.shape()[pruned_axis],
                        'target_len': in_var.shape()[pruned_axis],
                        'stride': 1
                    }
                    start += in_var.shape()[pruned_axis]

                    self._visit(in_var, pruned_axis)
                    pre_ops = in_var.inputs()
                    for op in pre_ops:
                        self._prune_op(op, in_var, pruned_axis,
                                       transforms + [transoform])
            else:
                for _, in_var in enumerate(self.op.inputs("X")):
                    self._visit(in_var, pruned_axis)
                    pre_ops = in_var.inputs()
                    for op in pre_ops:
                        self._prune_op(op, in_var, pruned_axis, transforms)
        elif var in self.op.inputs("X"):
            self._visit(var, pruned_axis)
            if axis == pruned_axis:
                idx = []
                target_start = 0
                for v in self.op.inputs("X"):
                    if v.name() != var.name():
                        target_start += v.shape()[pruned_axis]
                    else:
                        break
                target_end = target_start + v.shape()[pruned_axis]
                out_var = self.op.outputs("Out")[0]
                next_ops = out_var.outputs()

                transform = {
                    'src_start': 0,
                    'src_end': var.shape()[pruned_axis],
                    'target_start': target_start,
                    'target_end': target_end,
                    'target_len': out_var.shape()[pruned_axis],
                    'stride': 1
                }

                self._visit(out_var, pruned_axis)
                for op in next_ops:
                    # The output of concat can be visited repeatedly
                    c_visited = {}
                    self._prune_op(
                        op,
                        out_var,
                        pruned_axis,
                        transforms + [transform],
                        visited=c_visited)
                    # Add nodes searched from concat into global visited array.
                    self.visited.update(c_visited)


@PRUNE_WORKER.register
class depthwise_conv2d(PruneWorker):
    def __init__(self, op, pruned_params, visited={}):
        super(depthwise_conv2d, self).__init__(op, pruned_params, visited)

    def _prune(self, var, pruned_axis, transforms):
        assert var not in self.op.inputs(
            "Filter"), "Unsupport for pruning depthwise conv2d directly."
        assert var not in self.op.outputs(
            "Output"
        ), "Unsupport for pruning output of depthwise conv2d directly."
        data_format = self.op.attr("data_format")
        groups = self.op.attr("groups")
        channel_axis = 1
        if data_format == "NHWC":
            channel_axis = 3
        if var in self.op.inputs("Input"):
            assert pruned_axis == channel_axis, "The Input of conv2d can only be pruned at channel axis, but got {}".format(
                pruned_axis)

            groups = var.shape()[channel_axis]
            filter_var = self.op.inputs("Filter")[0]
            transform = {
                'src_start': 0,
                'src_end': var.shape()[pruned_axis],
                'target_start': 0,
                'target_end': filter_var.shape()[0],
                'target_len': filter_var.shape()[0],
                'stride': 1
            }

            self.pruned_params.append((filter_var, 0, transforms + [transform]))
            self._visit(filter_var, 0)

            for op in filter_var.outputs():
                self._prune_op(op, filter_var, 0, transforms + [transform])

            output_var = self.op.outputs("Output")[0]
            next_ops = output_var.outputs()
            for op in next_ops:
                self._prune_op(op, output_var, channel_axis,
                               transforms + [transform])


@PRUNE_WORKER.register
class mul(PruneWorker):
    def __init__(self, op, pruned_params, visited={}):
        super(mul, self).__init__(op, pruned_params, visited)

    def _prune(self, var, pruned_axis, pruned_idx):
        if var in self.op.inputs("X"):
            assert pruned_axis == 1, "The Input of conv2d can only be pruned at axis 1, but got {}".format(
                pruned_axis)
            idx = []
            feature_map_size = var.shape()[2] * var.shape()[3]
            range_idx = np.array(range(feature_map_size))
            for i in pruned_idx:
                idx += list(range_idx + i * feature_map_size)
            param_var = self.op.inputs("Y")[0]
            self.pruned_params.append((param_var, 0, idx))

            for op in param_var.outputs():
                self._prune_op(op, param_var, 0, pruned_idx)


@PRUNE_WORKER.register
class matmul(PruneWorker):
    def __init__(self, op, pruned_params, visited):
        super(matmul, self).__init__(op, pruned_params, visited)

    def _prune(self, var, pruned_axis, pruned_idx):
        if var in self.op.inputs("X") and pruned_axis == 1:
            param_var = self.op.inputs("Y")[0]
            self.pruned_params.append((param_var, 0, pruned_idx))

            for op in param_var.outputs():
                self._prune_op(op, param_var, 0, pruned_idx)


@PRUNE_WORKER.register
class scale(PruneWorker):
    def __init__(self, op, pruned_params, visited={}):
        super(scale, self).__init__(op, pruned_params, visited)

    def _prune(self, var, pruned_axis, pruned_idx):
        if var in self.op.inputs("X"):
            out_var = self.op.outputs("Out")[0]
            for op in out_var.outputs():
                self._prune_op(op, out_var, pruned_axis, pruned_idx)
        elif var in self.op.outputs("Out"):
            in_var = self.op.inputs("X")[0]
            for op in in_var.inputs():
                self._prune_op(op, in_var, pruned_axis, pruned_idx)


@PRUNE_WORKER.register
class momentum(PruneWorker):
    def __init__(self, op, pruned_params, visited={}):
        super(momentum, self).__init__(op, pruned_params, visited)

    def _prune(self, var, pruned_axis, pruned_idx):
        if var in self.op.inputs("Param"):
            _logger.debug("pruning momentum, var:{}".format(var.name()))
            velocity_var = self.op.inputs("Velocity")[0]
            self.pruned_params.append((velocity_var, pruned_axis, pruned_idx))


@PRUNE_WORKER.register
class adam(PruneWorker):
    def __init__(self, op, pruned_params, visited={}):
        super(adam, self).__init__(op, pruned_params, visited)

    def _prune(self, var, pruned_axis, pruned_idx):
        if var in self.op.inputs("Param"):
            _logger.debug("pruning momentum, var:{}".format(var.name()))
            moment1_var = self.op.inputs("Moment1")[0]
            self.pruned_params.append((moment1_var, pruned_axis, pruned_idx))
            moment2_var = self.op.inputs("Moment2")[0]
            self.pruned_params.append((moment2_var, pruned_axis, pruned_idx))


@PRUNE_WORKER.register
class affine_channel(PruneWorker):
    def __init__(self, op, pruned_params, visited):
        super(affine_channel, self).__init__(op, pruned_params, visited)

    def _prune(self, var, pruned_axis, pruned_idx):
        if (var not in self.op.outputs("Out")) and (
                var not in self.op.inputs("X")):
            return

        if var in self.op.outputs("Out"):
            in_var = self.op.inputs("X")[0]
            self._visit(in_var, pruned_axis)
            pre_ops = in_var.inputs()
            for op in pre_ops:
                self._prune_op(op, in_var, pruned_axis, pruned_idx)

        for param in ["Scale", "Bias"]:
            param_var = self.op.inputs(param)[0]
            for op in param_var.outputs():
                self._prune_op(op, param_var, 0, pruned_idx)
            self.pruned_params.append((param_var, 0, pruned_idx))

        out_var = self.op.outputs("Out")[0]
        self._visit(out_var, pruned_axis)
        next_ops = out_var.outputs()
        for op in next_ops:
            self._prune_op(op, out_var, pruned_axis, pruned_idx)


@PRUNE_WORKER.register
class flatten_contiguous_range(PruneWorker):
    def __init__(self, op, pruned_params, visited):
        super(flatten_contiguous_range, self).__init__(op, pruned_params,
                                                       visited)

    def _prune(self, var, pruned_axis, transforms):
        start_axis = self.op.attr("start_axis")
        stop_axis = self.op.attr("stop_axis")
        if var in self.op.inputs("X"):
            out_var = self.op.outputs("Out")[0]
            in_var = self.op.inputs("X")[0]
            stride = 1
            out_pruned_axis = pruned_axis
            if pruned_axis >= start_axis and pruned_axis <= stop_axis:
                out_pruned_axis = start_axis
                for i in range(pruned_axis + 1, stop_axis + 1):
                    stride *= in_var.shape()[i]
            elif pruned_axis > stop_axis:
                out_pruned_axis = start_axis + pruned_axis - stop_axis

            self._visit(in_var, pruned_axis)
            self._visit(out_var, out_pruned_axis)
            transform = {'stride': stride}
            next_ops = out_var.outputs()
            for op in next_ops:
                self._prune_op(op, out_var, out_pruned_axis,
                               transforms + [transform])
