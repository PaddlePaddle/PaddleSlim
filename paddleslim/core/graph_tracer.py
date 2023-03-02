# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
""" A structure used to describe the network of a model."""

import sys
import types
import paddle

from paddleslim.core import GraphWrapper
from .graph import Graph, Node
from paddleslim.core.dygraph import dygraph2program

__all__ = ["GraphTracer"]


def _apply(layer, func):
    for name, child in layer.named_children():
        func(child)
        _apply(child, func)


def _add_call_hook(module,
                   function_new,
                   method_name='forward',
                   backup_name='__forward_orig__'):
    def _call_hook_enable(op):
        # do not patch the top level modules. makes it easy to invoke by self.module(x)
        if op is not module:
            assert not hasattr(
                op, backup_name
            ), f'in {op.__class__.__name__} detected an existing function {backup_name} : please double check'
            # backup the original forward of op into backup_name
            method_orig = getattr(op, method_name)
            setattr(op, backup_name, method_orig)
            # set new method
            method_new = types.MethodType(function_new, op)
            setattr(op, method_name, method_new)

    _apply(module, _call_hook_enable)


def _remove_call_hook(module,
                      method_name='forward',
                      backup_name='__forward_orig__'):
    def _call_hook_disable(op):
        if op is not module:
            if hasattr(op, backup_name):
                method_new = getattr(op, method_name)
                method_orig = getattr(op, backup_name)
                setattr(op, method_name, method_orig)
                # delete the backup
                setattr(op, backup_name, method_new)
                delattr(op, backup_name)

    _apply(module, _call_hook_disable)


class GraphTracer(paddle.nn.Layer):
    """ A tool used to trace the execution of the model.
    Call the forward of the model decorated by this tracer
    and it will create a graph.

    Args:
        model(paddle.nn.Layer): The model to be traced.

    Examples:
       .. code-block:: python
            from paddeslim.core.graph_tracer import GraphTracer
            from paddle.vision.models import resnet18

            model = resnet18()
            x = paddle.rand([1, 3, 224, 224])
            tracer = GraphTracer(model)
            tracer(x)
            print(tracer.graph)

    """

    def __init__(self, model: paddle.nn.Layer):
        super(GraphTracer, self).__init__()
        self._model = model
        self._graph = None
        self._call_count = {}
        self._tensor_previous = {}

    @property
    def graph(self) -> Graph:
        assert self._graph is not None, "Please trace the graph by calling forward function of current tracer."
        return self._graph

    def forward(self, inputs, *args, **kwargs):
        self._graph = Graph()
        _add_call_hook(self._model, self._analyze_modules_op)
        self._model(inputs, *args, **kwargs)
        _remove_call_hook(self._model)

    def _analyze_modules_op(self, op, inputs, *args, **kwargs):
        node = self._trace_in(op, inputs)
        #print(f"inputs: {inputs.name}")
        outputs = op.__forward_orig__(inputs, *args, **kwargs)
        #print(f"outputs: {outputs.name}")
        self._trace_out(node, outputs)
        return outputs

    def _call_layer(self, layer):
        layer_name = layer.full_name()
        if layer_name not in self._call_count:
            self._call_count[layer_name] = 0
        self._call_count[layer_name] += 1
        return self._call_count[layer_name]

    def _trace_in(self, layer, inputs):
        inputs = self._convert_to_list(inputs)
        call_cout = self._call_layer(layer)
        current_node = Node(layer, call_cout)

        if current_node.name not in self._graph._name2node:
            self._graph._name2node[current_node.name] = current_node
        current_node = self._graph._name2node[current_node.name]
        for inp in inputs:
            last_node = self._tensor_previous.get(inp.name, None)
            if last_node is not None:
                assert isinstance(last_node, Node)
                if last_node not in current_node.previous_nodes:
                    current_node.previous_nodes.append(last_node)
                if current_node not in last_node.next_nodes:
                    last_node.next_nodes.append(current_node)

        return current_node

    def _trace_out(self, current_node, outputs):
        assert current_node is not None, "The current node has not been visited."
        if current_node.is_leaf():
            outputs = self._convert_to_list(outputs)
            for out in outputs:
                self._tensor_previous[out.name] = current_node

    def _convert_to_list(self, tensors):
        """ Convert tensor to list.
        It is important to convert the inputs to a list.
        Because visiting the tensor by 'for ... in' will create new
        temp variables and break the tracing process.
        """
        if isinstance(tensors, paddle.Tensor):
            return [tensors]
        elif isinstance(tensors, (list, tuple)):
            for _t in tensors:
                assert isinstance(_t, paddle.Tensor)
            return tensors
        raise TypeError(
            f"Unsopported type: {type(tensors)}; The inputs type should be paddle.Tensor' or list of paddle.Tensor."
        )
