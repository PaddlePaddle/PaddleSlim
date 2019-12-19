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

import numpy as np
from ..core import Registry

__all__ = ["PRUNE_WORKER", "conv2d"]

PRUNE_WORKER = Registry('prune_worker')

class PruneWorker(object):
    def __init__(self, op, pruned_params=[], visited={}):
        self.op = op
        self.pruned_params = pruned_params
        self.visited=visited

    def prune(self, var, pruned_aixs, pruned_idx):
        pass

    def prune_op(self, op, var, pruned_aixs, pruned_idx):
        if op.type().endswith("_grad"):
            return
        cls = PRUNE_WORKER.get(op.type())
        assert cls is not None , "The walker of {} is not registered.".format(op.type())
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
            assert pruned_axis == 1, "The Input of conv2d can only be pruned at axis 1, but got {}".format(pruned_axis)
            self.pruned_params.append((self.op.inputs("Filter")[0], 1, pruned_idx))
            
        elif var in self.op.inputs("Filter"):
            assert pruned_axis in [0,1]
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
            assert pruned_axis == 1
            self.pruned_params.append((self.op.inputs("Filter")[0], 0, pruned_idx))
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

        if var in self.op.outputs("Y"):
            in_var = self.op.inputs("X")[0]
            key = "_".join([str(self.op.idx()), in_var.name()])
            self.visited[pruned_axis][key] = True
            pre_ops = in_var.inputs()
            for op in pre_ops:
                self.prune_op(op, in_var, pruned_axis, pruned_idx)
        for param in ["Scale", "Bias", "Mean", "Variance"]: 
            self.pruned_params.append((self.op.inputs(param)[0], 0, pruned_idx))

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
        if var in self.op.outputs("Out"):
            for name in ["X", "Y"]:
                in_var = self.op.inputs(name)[0]
                pre_ops = in_var.inputs()
                for op in pre_ops:
                    self.prune_op(op, in_var, pruned_axis, pruned_idx)

        else:
            if var in self.op.inputs("X"):
                in_var = self.op.inputs("Y")[0]
                pre_ops = in_var.inputs()
                for op in pre_ops:
                    self.prune_op(op, in_var, pruned_axis, pruned_idx)
            elif var in self.op.inputs("Y"):
                in_var = self.op.inputs("X")[0]
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

    def prune(self, var, pruned_axis, pruned_idx):
        key = "_".join([str(self.op.idx()), var.name()])
        if pruned_axis not in self.visited:
            self.visited[pruned_axis] = {}
        if key in self.visited[pruned_axis]:
            return
        else:
           self.visited[pruned_axis][key] = True
        
        if var in self.op.outputs("Out"):
            in_var = self.op.inputs("X")[0]
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
class relu(activation):
    def __init__(self, op, pruned_params, visited):
        super(relu, self).__init__(op, pruned_params, visited)

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
            self.pruned_params.append((self.op.inputs("Filter")[0], 0, pruned_idx))

            output_var = self.op.outputs("Output")[0]
            next_ops = output_var.outputs()
            for op in next_ops:
                self.prune_op(op, output_var, 1, pruned_idx)
            
        elif var in self.op.inputs("Filter"):
            assert pruned_axis in [0]
            if pruned_axis == 0:
                if len(self.op.inputs("Bias")) > 0:                
                    self.pruned_params.append((self.op.inputs("Bias"), 1, pruned_idx))
                output_var = self.op.outputs("Output")[0]
                key = "_".join([str(self.op.idx()), output_var.name()])
                self.visited[pruned_axis][key] = True
                next_ops = output_var.outputs()
                for op in next_ops:
                    self.prune_op(op, output_var, 1, pruned_idx)

#            elif pruned_axis == 1:
#                input_var = self.op.inputs("Input")[0]
#                key = "_".join([str(self.op.idx()), input_var.name()])
#                self.visited[pruned_axis][key] = True
#                pre_ops = input_var.inputs()
#                for op in pre_ops:
#                    self.prune_op(op, input_var, 1, pruned_idx)
        elif var in self.op.outputs("Output"):
            assert pruned_axis == 1
            self.pruned_params.append((self.op.inputs("Filter")[0], 0, pruned_idx))
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
            self.pruned_params.append((self.op.inputs("Y")[0], 0, idx))
            

