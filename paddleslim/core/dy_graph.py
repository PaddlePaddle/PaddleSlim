# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import copy
import pickle
import numpy as np
from collections import OrderedDict
from collections import Iterable
import torch

__all__ = ['DyGraph', 'VarWrapper', 'OpWrapper']


class VarWrapper(object):
    def __init__(self, id, is_parameter=False, tensor=None):
        self._id = id
        self._inputs = []
        self._outputs = []
        self._is_parameter = is_parameter
        self._tensor = tensor

    def __eq__(self, v):
        """
        Overwrite this function for ...in... syntax in python.
        """
        return self._id == v._id

    def name(self):
        """
        Get the name of the variable.
        """
        return self._id

    def __repr__(self):
        return "id: {};".format(self._id)

    def shape(self):
        """
        Get the shape of the varibale.
        """
        return self._tensor.shape

    def set_shape(self, shape):
        """
        Set the shape of the variable.
        """
        assert ("Unimplement")

    def inputs(self):
        """
        Get all the operators that use this variable as output.

        Returns:
            list<OpWrapper>: A list of operators.
        """
        return self._inputs

    def outputs(self):
        """
        Get all the operators that use this variable as input.

        Returns:
            list<OpWrapper>: A list of operators.
        """
        return self._outputs

    def is_parameter(self):
        return self._is_parameter


class OpWrapper(object):
    def __init__(self, id, name):
        self._id = id
        self.name = name
        self.module = None
        self._inputs = []
        self._outputs = []

    def __eq__(self, op):
        """
        Overwrite this function for ...in... syntax in python.
        """
        return self.id() == op.id()

    def all_inputs(self):
        """
        Get all the input variables of this operator.
        """
        return self._inputs

    def all_outputs(self):
        """
        Get all the output variables of this operator.
        """
        return self._outputs

    def id(self):
        """
        Get the id of this operator.
        """
        return self._id

    def type(self):
        """
        Get the type of this operator.
        """
        if self.module is not None:
            return self.module.__class__.__name__
        else:
            if self.name.startswith("aten::"):
                return self.name.split(":")[-1]

    def __repr__(self):
        return "op[id: {}, type: {}; inputs: {}]".format(self.id(),
                                                         self.type(),
                                                         self.all_inputs())

    def is_bwd_op(self):
        """
        Whether this operator is backward op.
        """
        return False

    def is_opt_op(self):
        """
        Whether this operator is optimizer op.
        """
        return False

    def inputs(self, name):
        """
        Get all the varibales by the input name.
        """
        return [self._graph.var(var_name) for var_name in self._op.input(name)]

    def outputs(self, name):
        """
        Get all the varibales by the output name.
        """
        return [
            self._graph.var(var_name) for var_name in self._op.output(name)
        ]

    def set_attr(self, key, value):
        """
        Set the value of attribute by attribute's name.

        Args:
            key(str): the attribute name.
            value(bool|int|str|float|list): the value of the attribute.
        """
        self._op._set_attr(key, value)

    def attr(self, name):
        """
        Get the attribute by name.

        Args:
            name(str): the attribute name.

        Returns:
            bool|int|str|float|list: The attribute value. The return value
            can be any valid attribute type.
        """
        print dir(self.module)
        return self._op.attr(name)


class DyGraph(object):
    """
    It is a wrapper of paddle.fluid.framework.IrGraph with some special functions
    for paddle slim framework.

    Args:
        program(framework.Program): A program with 
        in_nodes(dict): A dict to indicate the input nodes of the graph.
                        The key is user-defined and human-readable name.
                        The value is the name of Variable.
        out_nodes(dict): A dict to indicate the input nodes of the graph.
                        The key is user-defined and human-readable name.
                        The value is the name of Variable.
    """

    def __init__(self, module, input_shape):
        """
        """
        super(DyGraph, self).__init__()
        self.module = module
        self._graph = torch.jit.trace(self.module,
                                      torch.rand(input_shape)).graph
        print self._graph
        self.children = {}
        for name, child in self.module.named_children():
            self.children[name] = child

        self.id2child = {}
        for node in self._graph.nodes():
            if "prim::GetAttr" == node.kind() and "self.1" == node.inputsAt(
                    0).debugName():
                #                    print dir(node)
                self.id2child[node.output().debugName()] = node["name"]

        print self.id2child

        self.vars = {}
        self.nodes = {}
        for node in self._graph.nodes():
            if "prim::CallMethod" == node.kind() and "forward" == node["name"]:
                module_id = node.inputsAt(0).debugName()
                node_id = node.output().debugName() + "-" + module_id
                in_var_id = node.inputsAt(1).debugName()
                out_var_id = node.output().debugName()
                if node_id not in self.nodes:
                    self.nodes[node_id] = OpWrapper(node_id,
                                                    self.id2child[module_id])
                    self.nodes[node_id].module = self.children[self.id2child[
                        module_id]]

                for param_id, param in self.nodes[
                        node_id].module.named_parameters():
                    param_id = ".".join([self.id2child[module_id], param_id])
                    if param_id not in self.vars:
                        self.vars[param_id] = VarWrapper(
                            param_id, is_parameter=True, tensor=param)
                        self.nodes[node_id].all_inputs().append(self.vars[
                            param_id])
                        self.vars[param_id].outputs().append(self.nodes[
                            node_id])

                if in_var_id not in self.vars:
                    self.vars[in_var_id] = VarWrapper(in_var_id)
                if out_var_id not in self.vars:
                    self.vars[out_var_id] = VarWrapper(out_var_id)
                self.nodes[node_id].all_inputs().append(self.vars[in_var_id])
                self.nodes[node_id].all_outputs().append(self.vars[out_var_id])
                self.vars[in_var_id].outputs().append(self.nodes[node_id])
                self.vars[out_var_id].inputs().append(self.nodes[node_id])
            elif node.kind().startswith("aten::"):
                #                print dir(node)
                node_id = node.output().debugName() + "-" + node.kind()
                #                node_id = node.debugName()
                if node_id not in self.nodes:
                    self.nodes[node_id] = OpWrapper(node_id, node.kind())

#                    self.nodes[node_id].type = node.kind()
                for input in node.inputs():
                    in_var_id = input.debugName()
                    if in_var_id not in self.vars:
                        self.vars[in_var_id] = VarWrapper(in_var_id)
                    self.vars[in_var_id].outputs().append(self.nodes[node_id])
                    self.nodes[node_id].all_inputs().append(self.vars[
                        in_var_id])

                for output in node.outputs():
                    out_var_id = output.debugName()
                    if out_var_id not in self.vars:
                        self.vars[out_var_id] = VarWrapper(out_var_id)
                    self.vars[out_var_id].inputs().append(self.nodes[node_id])
                    self.nodes[node_id].all_outputs().append(self.vars[
                        out_var_id])

    def all_parameters(self):
        """
        Get all the parameters in this graph.

        Returns:
            list<VarWrapper>: A list of VarWrapper instances.
        """
        params = []
        for var in self.vars.values():
            if var.is_parameter():
                params.append(var)
        return params

    def is_parameter(self, var):
        """
        Whether the given variable is parameter.

        Args:
            var(VarWrapper): The given varibale.
        """
        return var.is_parameter()

    def ops(self):
        """
        Return all operator nodes included in the graph as a set.
        """
        return self.nodes.values()

    def vars(self):
        """
        Get all the variables.
        """
        return self.vars.values()

    def var(self, name):
        """
        Get the variable by variable name.
        """
        return self.vars[name]

    def clone(self, for_test=False):
        """
        Clone a new graph from current graph.

        Returns:
            (DyGraph): The wrapper of a new graph.
        """
        return DyGraph(
            self.program.clone(for_test),
            copy.deepcopy(self.in_nodes), copy.deepcopy(self.out_nodes))

    def program(self):
        """
        Get the program in current wrapper.
        """
        return self.program

    def pre_ops(self, op):
        """
        Get all the previous operators of target operator.

        Args:
            op(OpWrapper): Target operator.

        Returns:
            list<OpWrapper>: A list of operators.
        """
        ops = []
        for p in self.ops():
            for in_var in op.all_inputs():
                if in_var in p.all_outputs():
                    ops.append(p)
        return ops

    def next_ops(self, op):
        """
        Get all the next operators of target operator.

        Args:
            op(OpWrapper): Target operator.

        Returns:
            list<OpWrapper>: A list of operators.
        """
        ops = []
        for p in self.ops():
            for out_var in op.all_outputs():
                if out_var in p.all_inputs():
                    ops.append(p)
        return ops

    def get_param_by_op(self, op):
        """
        Get the parameters used by target operator.
        """
        assert isinstance(op, OpWrapper)
        params = []
        for var in op.all_inputs():
            if isinstance(var._var, Parameter):
                params.append(var)
        assert len(params) > 0
        return params

    def numel_params(self):
        """
        Get the number of elements in all parameters.
        """
        ret = 0
        for param in self.all_parameters():
            ret += np.product(param.shape())
        return ret

    def update_param_shape(self, scope):
        """
        Update the shape of parameters in the graph according to tensors in scope.
        It is used after loading pruned parameters from file.
        """
        for param in self.all_parameters():
            tensor_shape = np.array(
                scope.find_var(param.name()).get_tensor()).shape
            param.set_shape(tensor_shape)

    def infer_shape(self):
        """
        Update the groups of convolution layer according to current filters.
        It is used after loading pruned parameters from file.
        """
        for op in self.ops():
            if op.type() != 'conditional_block':
                op._op.desc.infer_shape(op._op.block.desc)

    def update_groups_of_conv(self):
        for op in self.ops():
            if op.type() == 'depthwise_conv2d' or op.type(
            ) == 'depthwise_conv2d_grad':
                op.set_attr('groups', op.inputs('Filter')[0].shape()[0])
