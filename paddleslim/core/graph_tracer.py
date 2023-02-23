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
"""A structure used to describe the network of a model."""

import types
import paddle
from .graph import Graph, Node

__all__ = ["GraphTracer"]


def apply(layer, func):
    for name, child in layer.named_children():
        func(child)
        apply(child, func)


def add_call_hook(module,
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

    apply(module, _call_hook_enable)


def remove_call_hook(module,
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

    apply(module, _call_hook_disable)


class GraphTracer(paddle.nn.Layer):
    def __init__(self, model: paddle.nn.Layer):
        super(GraphTracer, self).__init__()
        self._model = model
        self._graph = None
        self._call_count = {}
        self._tensor_previous = {}

    @property
    def graph(self):
        assert self._graph is not None, "Please trace the graph by calling forward function of current tracer."
        return self._graph

    def forward(self, inputs, *args, **kwargs):
        self._graph = Graph()
        add_call_hook(self._model, self._analyze_modules_op)
        self._model(inputs, *args, **kwargs)
        remove_call_hook(self._model)

    def _analyze_modules_op(self, op, inputs, *args, **kwargs):
        node = self.trace_in(op, inputs)
        outputs = op.__forward_orig__(inputs, *args, **kwargs)
        self.trace_out(node, inputs)
        return outputs

    def _call_layer(self, layer):
        layer_name = layer.full_name()
        if layer_name not in self._call_count:
            self._call_count[layer_name] = 0
        self._call_count[layer_name] += 1
        return self._call_count[layer_name]

    def trace_in(self, layer, inputs):
        call_cout = self._call_layer(layer)
        current_node = Node(layer, call_cout)

        if current_node.name not in self._graph._name2node:
            self._graph._name2node[current_node.name] = current_node
        current_node = self._graph._name2node[current_node.name]

        for inp in inputs:
            last_node = self._tensor_previous.get(hash(inp), None)
            if last_node is not None:
                assert isinstance(last_node, Node)
                if last_node not in current_node._previous_node:
                    current_node._previous_node.append(last_node)
                if current_node not in last_node._next_node:
                    last_node._next_node.append(current_node)

        return current_node

    def is_leaf(self, layer):
        return isinstance(layer,
                          paddle.nn.Layer) and len(layer.sublayers()) == 0

    def trace_out(self, current_node, outputs):
        assert current_node is not None, "The current node has not been visited."
        if current_node.is_leaf():
            for out in outputs:
                self._tensor_previous[hash(out)] = current_node
