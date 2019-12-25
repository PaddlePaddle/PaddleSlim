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

class PruneWorker(object):
    def __init__(self, op, pruned_params=[], visited={}):
        self.op = op
        self.pruned_params = pruned_params
        self.visited=visited

    def prune(self, var, pruned_aixs, pruned_idx):
        pass

    def prune_op(self, op, var, pruned_aixs, pruned_idx, visited=None):
        if op.type().endswith("_grad"):
            return
        if visited is not None:
            self.visited = visited
        cls = PRUNE_WORKER.get(op.type())
        assert cls is not None , "The walker of {} is not registered.".format(op.type())
        _logger.debug("\nfrom: {}\nto: {}\npruned_aixs: {}; var: {}".format(self.op, op, pruned_aixs, var.name()))
        walker = cls(op, pruned_params=self.pruned_params, visited=self.visited)
        walker.prune(var, pruned_aixs, pruned_idx)

@PRUNE_WORKER.register
class conv2d(PruneWorker):

    def __init__(self, op, pruned_params, visited={}):
        super(conv2d, self).__init__(op, pruned_params, visited)

    def prune(self, var, pruned_axis, pruned_idx):
        key = "_".join([str(self.op.idx()), var.name()])
        if pruned_axis not in self.visited:
            self.visited[pruned_axis] = {}
        if key in self.visited[pruned_axis]:
            return
        else:
           self.visited[pruned_axis][key] = True

        if var in self.op.inputs("Input"):
            assert pruned_axis == 1, "The Input of conv2d can only be pruned at axis 1, but got {}; var: {}".format(pruned_axis, var.name())
            filter_var = self.op.inputs("Filter")[0]
            key = "_".join([str(self.op.idx()), filter_var.name()])
            self.visited[1][key] = True
            self.pruned_params.append((filter_var, 1, pruned_idx))
            for op in filter_var.outputs():
                self.prune_op(op, filter_var, 1, pruned_idx)
             
        elif var in self.op.inputs("Filter"):
            assert pruned_axis in [0,1]

            self.pruned_params.append((var, pruned_axis, pruned_idx))

            for op in var.outputs():
                self.prune_op(op, var, pruned_axis, pruned_idx)

            if pruned_axis == 0:
                if len(self.op.inputs("Bias")) > 0:                
                    self.pruned_params.append((self.op.inputs("Bias"), 1, pruned_idx))
                output_var = self.op.outputs("Output")[0]
                key = "_".join([str(self.op.idx()), output_var.name()])
                self.visited[pruned_axis][key] = True
                next_ops = output_var.outputs()
                for op in next_ops:
                    self.prune_op(op, output_var, 1, pruned_idx)

            elif pruned_axis == 1:
                input_var = self.op.inputs("Input")[0]
                key = "_".join([str(self.op.idx()), input_var.name()])
                self.visited[pruned_axis][key] = True
                pre_ops = input_var.inputs()
                for op in pre_ops:
                    self.prune_op(op, input_var, 1, pruned_idx)
        elif var in self.op.outputs("Output"):
            assert pruned_axis == 1, "pruned_axis: {}; var: {}".format(pruned_axis, var.name())

            filter_var = self.op.inputs("Filter")[0]
            key = "_".join([str(self.op.idx()), filter_var.name()])
            self.visited[0][key] = True

            self.pruned_params.append((filter_var, 0, pruned_idx))

            for op in filter_var.outputs():
                self.prune_op(op, filter_var, 0, pruned_idx)

            if len(self.op.inputs("Bias")) > 0:                
                self.pruned_params.append((self.op.inputs("Bias")[0], 1, pruned_idx))
                
            output_var = self.op.outputs("Output")[0]
            next_ops = output_var.outputs()
            for op in next_ops:
                self.prune_op(op, output_var, 1, pruned_idx)
            

@PRUNE_WORKER.register
class batch_norm(PruneWorker):

    def __init__(self, op, pruned_params, visited):
        super(batch_norm, self).__init__(op, pruned_params, visited)

    def prune(self, var, pruned_axis, pruned_idx):
        key = "_".join([str(self.op.idx()), var.name()])
        if pruned_axis not in self.visited:
            self.visited[pruned_axis] = {}
        if key in self.visited[pruned_axis]:
            return
        else:
           self.visited[pruned_axis][key] = True

        if (var not in self.op.outputs("Y")) and (var not in self.op.inputs("X")):
            return

        if var in self.op.outputs("Y"):
            in_var = self.op.inputs("X")[0]
            key = "_".join([str(self.op.idx()), in_var.name()])
            self.visited[pruned_axis][key] = True
            pre_ops = in_var.inputs()
            for op in pre_ops:
                self.prune_op(op, in_var, pruned_axis, pruned_idx)
        
        for param in ["Scale", "Bias", "Mean", "Variance"]: 
            param_var = self.op.inputs(param)[0]
            for op in param_var.outputs():
                self.prune_op(op, param_var, 0, pruned_idx)
            self.pruned_params.append((param_var, 0, pruned_idx))
            

        out_var = self.op.outputs("Y")[0]
        key = "_".join([str(self.op.idx()), out_var.name()])
        self.visited[pruned_axis][key] = True
        next_ops = out_var.outputs()
        for op in next_ops:
            self.prune_op(op, out_var, pruned_axis, pruned_idx)


class elementwise_op(PruneWorker):
    def __init__(self, op, pruned_params, visited):
        super(elementwise_op, self).__init__(op, pruned_params, visited)
    
    def prune(self, var, pruned_axis, pruned_idx):
        key = "_".join([str(self.op.idx()), var.name()])
        if pruned_axis not in self.visited:
            self.visited[pruned_axis] = {}
        if key in self.visited[pruned_axis]:
            return
        else:
           self.visited[pruned_axis][key] = True
        axis = self.op.attr("axis")
        if axis == -1: # TODO
            axis = 0
        if var in self.op.outputs("Out"):
            for name in ["X", "Y"]:
                actual_axis = pruned_axis
                if name == "Y":
                    actual_axis = pruned_axis - axis
                in_var = self.op.inputs(name)[0]
                pre_ops = in_var.inputs()
                for op in pre_ops:
                    self.prune_op(op, in_var, actual_axis, pruned_idx)

        else:
            if var in self.op.inputs("X"):
                in_var = self.op.inputs("Y")[0]

                if in_var.is_parameter():
                    self.pruned_params.append((in_var, pruned_axis - axis, pruned_idx))
                pre_ops = in_var.inputs()
                for op in pre_ops:
                    self.prune_op(op, in_var, pruned_axis - axis, pruned_idx)
            elif var in self.op.inputs("Y"):
                in_var = self.op.inputs("X")[0]
                pre_ops = in_var.inputs()
                pruned_axis = pruned_axis + axis
                for op in pre_ops:
                    self.prune_op(op, in_var, pruned_axis, pruned_idx)
    
        out_var = self.op.outputs("Out")[0]
        key = "_".join([str(self.op.idx()), out_var.name()])
        self.visited[pruned_axis][key] = True
        next_ops = out_var.outputs()
        for op in next_ops:
            self.prune_op(op, out_var, pruned_axis, pruned_idx)


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

class activation(PruneWorker):

    def __init__(self, op, pruned_params, visited):
        super(activation, self).__init__(op, pruned_params, visited)
        self.input_name = "X"
        self.output_name = "Out"

    def prune(self, var, pruned_axis, pruned_idx):
        key = "_".join([str(self.op.idx()), var.name()])
        if pruned_axis not in self.visited:
            self.visited[pruned_axis] = {}
        if key in self.visited[pruned_axis]:
            return
        else:
           self.visited[pruned_axis][key] = True
        
        if var in self.op.outputs(self.output_name):
            in_var = self.op.inputs(self.input_name)[0]
            pre_ops = in_var.inputs()
            for op in pre_ops:
                self.prune_op(op, in_var, pruned_axis, pruned_idx)


        out_var = self.op.outputs(self.output_name)[0]
        key = "_".join([str(self.op.idx()), out_var.name()])
        self.visited[pruned_axis][key] = True
        next_ops = out_var.outputs()
        for op in next_ops:
            self.prune_op(op, out_var, pruned_axis, pruned_idx)



@PRUNE_WORKER.register
class uniform_random_batch_size_like(activation):
    def __init__(self, op, pruned_params, visited):
        super(uniform_random_batch_size_like, self).__init__(op, pruned_params, visited)
        self.input_name = "Input"
        self.output_name = "Out"

@PRUNE_WORKER.register
class bilinear_interp(activation):
    def __init__(self, op, pruned_params, visited):
        super(bilinear_interp, self).__init__(op, pruned_params, visited)

@PRUNE_WORKER.register
class relu(activation):
    def __init__(self, op, pruned_params, visited):
        super(relu, self).__init__(op, pruned_params, visited)

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
    
    def prune(self, var, pruned_axis, pruned_idx):
        key = "_".join([str(self.op.idx()), var.name()])
        if pruned_axis not in self.visited:
            self.visited[pruned_axis] = {}
        if key in self.visited[pruned_axis]:
            return
        else:
           self.visited[pruned_axis][key] = True

        if var in self.op.outputs("Out"):
             for in_var in self.op.inputs("X"):
                pre_ops = in_var.inputs()
                for op in pre_ops:
                    self.prune_op(op, in_var, pruned_axis, pruned_idx)
        elif var in self.op.inputs("X"):
            for in_var in self.op.inputs("X"):
                if in_var != var:
                    pre_ops = in_var.inputs()
                    for op in pre_ops:
                        self.prune_op(op, in_var, pruned_axis, pruned_idx)
        out_var = self.op.outputs("Out")[0]
        key = "_".join([str(self.op.idx()), out_var.name()])
        self.visited[pruned_axis][key] = True
        next_ops = out_var.outputs()
        for op in next_ops:
            self.prune_op(op, out_var, pruned_axis, pruned_idx)

@PRUNE_WORKER.register
class concat(PruneWorker):
    def __init__(self, op, pruned_params, visited):
        super(concat, self).__init__(op, pruned_params, visited)
    
    def prune(self, var, pruned_axis, pruned_idx):
        key = "_".join([str(self.op.idx()), var.name()])
        if pruned_axis not in self.visited:
            self.visited[pruned_axis] = {}
        if key in self.visited[pruned_axis]:
            return
        else:
           self.visited[pruned_axis][key] = True
        idx = []
        axis = self.op.attr("axis")
        if var in self.op.outputs("Out"):
             start = 0
             if axis == pruned_axis: 
                 for _, in_var in enumerate(self.op.inputs("X")):
                    idx = []
                    for i in pruned_idx:
                        r_idx = i - start
                        if r_idx < in_var.shape()[pruned_axis] and r_idx >= 0:
                            idx.append(r_idx)
                    start += in_var.shape()[pruned_axis]
                    
                    pre_ops = in_var.inputs()
                    for op in pre_ops:
                        self.prune_op(op, in_var, pruned_axis, idx)
                 idx = pruned_idx[:]
             else:
                 for _, in_var in enumerate(self.op.inputs("X")):
                     pre_ops = in_var.inputs()
                     for op in pre_ops:
                         self.prune_op(op, in_var, pruned_axis, pruned_idx)
        elif var in self.op.inputs("X"):
            if axis == pruned_axis: 
                idx = []
                start = 0
                for v in self.op.inputs("X"):
                    if v.name() == var.name():
                        idx = [i+start for i in pruned_idx]
                    else:
                        start += v.shape()[pruned_axis]
    
                out_var = self.op.outputs("Out")[0]
                key = "_".join([str(self.op.idx()), out_var.name()])
                self.visited[pruned_axis][key] = True
                next_ops = out_var.outputs()
                for op in next_ops:
                    self.prune_op(op, out_var, pruned_axis, idx, visited={})
            else:
                for v in self.op.inputs("X"):
                    for op in v.inputs():
                        self.prune_op(op, v, pruned_axis, pruned_idx)
                out_var = self.op.outputs("Out")[0]
                key = "_".join([str(self.op.idx()), out_var.name()])
                self.visited[pruned_axis][key] = True
                next_ops = out_var.outputs()
                for op in next_ops:
                    self.prune_op(op, out_var, pruned_axis, pruned_idx)


@PRUNE_WORKER.register
class depthwise_conv2d(PruneWorker):

    def __init__(self, op, pruned_params, visited={}):
        super(depthwise_conv2d, self).__init__(op, pruned_params, visited)

    def prune(self, var, pruned_axis, pruned_idx):
        key = "_".join([str(self.op.idx()), var.name()])
        if pruned_axis not in self.visited:
            self.visited[pruned_axis] = {}
        if key in self.visited[pruned_axis]:
            return
        else:
           self.visited[pruned_axis][key] = True

        if var in self.op.inputs("Input"):
            assert pruned_axis == 1, "The Input of conv2d can only be pruned at axis 1, but got {}".format(pruned_axis)

            filter_var = self.op.inputs("Filter")[0]
            self.pruned_params.append((filter_var, 0, pruned_idx))
            key = "_".join([str(self.op.idx()), filter_var.name()])
            self.visited[0][key] = True

            new_groups = filter_var.shape()[0] - len(pruned_idx)
            self.op.set_attr("groups", new_groups)

            for op in filter_var.outputs():
                self.prune_op(op, filter_var, 0, pruned_idx)

            output_var = self.op.outputs("Output")[0]
            next_ops = output_var.outputs()
            for op in next_ops:
                self.prune_op(op, output_var, 1, pruned_idx)
            
        elif var in self.op.inputs("Filter"):
            assert pruned_axis in [0]
            if pruned_axis == 0:
                if len(self.op.inputs("Bias")) > 0:                
                    self.pruned_params.append((self.op.inputs("Bias"), 1, pruned_idx))

                self.pruned_params.append((var, 0, pruned_idx))
                new_groups = var.shape()[0] - len(pruned_idx)
                self.op.set_attr("groups", new_groups)

                for op in var.outputs():
                    self.prune_op(op, var, 0, pruned_idx)

                output_var = self.op.outputs("Output")[0]
                key = "_".join([str(self.op.idx()), output_var.name()])
                self.visited[pruned_axis][key] = True
                next_ops = output_var.outputs()
                for op in next_ops:
                    self.prune_op(op, output_var, 1, pruned_idx)
            for op in var.outputs():
                self.prune_op(op, var, pruned_axis, pruned_idx)
        elif var in self.op.outputs("Output"):
            assert pruned_axis == 1
            filter_var = self.op.inputs("Filter")[0]
            self.pruned_params.append((filter_var, 0, pruned_idx))
            key = "_".join([str(self.op.idx()), filter_var.name()])
            self.visited[0][key] = True

            new_groups = filter_var.shape()[0] - len(pruned_idx)
            op.set_attr("groups", new_groups)

            for op in filter_var.outputs():
                self.prune_op(op, filter_var, 0, pruned_idx)

            if len(self.op.inputs("Bias")) > 0:                
                self.pruned_params.append((self.op.inputs("Bias")[0], 1, pruned_idx))

            in_var = self.op.inputs("Input")[0]
            key = "_".join([str(self.op.idx()), in_var.name()])
            self.visited[pruned_axis][key] = Tru
            pre_ops = in_var.inputs()
            for op in pre_ops:
                self.prune_op(op, in_var, 1, pruned_idx)

            output_var = self.op.outputs("Output")[0]
            next_ops = output_var.outputs()
            for op in next_ops:
                self.prune_op(op, output_var, 1, pruned_idx)
        

@PRUNE_WORKER.register
class mul(PruneWorker):

    def __init__(self, op, pruned_params, visited={}):
        super(mul, self).__init__(op, pruned_params, visited)

    def prune(self, var, pruned_axis, pruned_idx):
        key = "_".join([str(self.op.idx()), var.name()])
        if pruned_axis not in self.visited:
            self.visited[pruned_axis] = {}
        if key in self.visited[pruned_axis]:
            return
        else:
           self.visited[pruned_axis][key] = True

        if var in self.op.inputs("X"):
            assert pruned_axis == 1, "The Input of conv2d can only be pruned at axis 1, but got {}".format(pruned_axis)
            idx = []
            feature_map_size = var.shape()[2] * var.shape()[3]
            range_idx = np.array(range(feature_map_size))
            for i in pruned_idx:
                idx += list(range_idx + i * feature_map_size) 
            param_var = self.op.inputs("Y")[0]
            self.pruned_params.append((param_var, 0, idx))

            for op in param_var.outputs():
                self.prune_op(op, param_var, 0, pruned_idx)
            

@PRUNE_WORKER.register
class scale(PruneWorker):
    def __init__(self, op, pruned_params, visited={}):
        super(scale, self).__init__(op, pruned_params, visited)

    def prune(self, var, pruned_axis, pruned_idx):
        key = "_".join([str(self.op.idx()), var.name()])
        if pruned_axis not in self.visited:
            self.visited[pruned_axis] = {}
        if key in self.visited[pruned_axis]:
            return
        else:
           self.visited[pruned_axis][key] = True

        if var in self.op.inputs("X"):
            out_var  = self.op.outputs("Out")[0]
            for op in out_var.outputs():
                self.prune_op(op, out_var, pruned_axis, pruned_idx)
        elif var in self.op.outputs("Out"):
            in_var  = self.op.inputs("X")[0]
            for op in in_var.inputs():
                self.prune_op(op, in_var, pruned_axis, pruned_idx)

@PRUNE_WORKER.register
class momentum(PruneWorker):
    def __init__(self, op, pruned_params, visited={}):
        super(momentum, self).__init__(op, pruned_params, visited)

    def prune(self, var, pruned_axis, pruned_idx):
    
        key = "_".join([str(self.op.idx()), var.name()])
        if pruned_axis not in self.visited:
            self.visited[pruned_axis] = {}
        if key in self.visited[pruned_axis]:
            return
        else:
            self.visited[pruned_axis][key] = True

        if var in self.op.inputs("Param"):
            _logger.debug("pruning momentum, var:{}".format(var.name()))
            velocity_var = self.op.inputs("Velocity")[0]
            self.pruned_params.append((velocity_var, pruned_axis, pruned_idx))

@PRUNE_WORKER.register
class adam(PruneWorker):
    def __init__(self, op, pruned_params, visited={}):
        super(adam, self).__init__(op, pruned_params, visited)

    def prune(self, var, pruned_axis, pruned_idx):
    
        key = "_".join([str(self.op.idx()), var.name()])
        if pruned_axis not in self.visited:
            self.visited[pruned_axis] = {}
        if key in self.visited[pruned_axis]:
            return
        else:
            self.visited[pruned_axis][key] = True

        if var in self.op.inputs("Param"):
            _logger.debug("pruning momentum, var:{}".format(var.name()))
            moment1_var = self.op.inputs("Moment1")[0]
            self.pruned_params.append((moment1_var, pruned_axis, pruned_idx))
            moment2_var = self.op.inputs("Moment2")[0]
            self.pruned_params.append((moment2_var, pruned_axis, pruned_idx))
