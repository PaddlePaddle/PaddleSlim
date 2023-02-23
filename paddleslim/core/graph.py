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

import paddle

__all__ = ["GraphTracer"]


class Node():
    def __init__(self, layer, call_count):
        self._layer = layer
        self._layer_name = layer.full_name()
        self._call_count = call_count
        self._hash_name = f"{self._layer_name}_{self._call_count}"
        self._next_node = []
        self._previous_node = []

    @property
    def name(self):
        return self._hash_name

    def is_leaf(self):
        return isinstance(self._layer,
                          paddle.nn.Layer) and len(self._layer.sublayers()) == 0

    def __hash__(self):
        return self._hash_name

    def __str__(self):
        return self._hash_name


class Graph():
    def __init__(self):
        # hash name to node
        self._name2node = {}

    def __str__(self):
        return str(self._name2node.keys())

    def find_conv_bn(self):
        results = []
        for name, node in self._name2node.items():
            if type(node._layer) in [paddle.nn.Conv2D]:
                for _next_node in node._next_node:
                    if type(_next_node._layer) in [paddle.nn.BatchNorm2D]:
                        results.append((node, _next_node))
                        break
        return results
