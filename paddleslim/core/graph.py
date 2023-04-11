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

from typing import List, Tuple
import paddle

__all__ = ["Graph"]


class Node():
    """ It is the node of the model's executing DAG. A node is a layer's once-calling.
    Args:
        layer(paddle.nn.Layer): The layer executed in the current node.
    """

    def __init__(self, layer, call_count):
        self._layer = layer
        self._layer_name = layer.full_name()
        self._call_count = call_count
        self._hash_name = f"{self._layer_name}_{self._call_count}"
        self._next_nodes = []
        self._previous_nodes = []

    @property
    def next_nodes(self):
        """ Get the next nodes of the current node.
        
        Returns:
            A list of nodes representing the next nodes of the current node.

        """
        return self._next_nodes

    @property
    def previous_nodes(self):
        """ Get the previous nodes of the current node.
        
        Returns:
            A list of nodes representing the previous nodes of the current node.
        """
        return self._previous_nodes

    @property
    def name(self):
        return self._hash_name

    @property
    def layer_name(self) -> str:
        """ Get the full name of the layer executed in the current node."""
        return self._layer_name

    @property
    def layer(self) -> paddle.nn.Layer:
        """ Get the layer executed in the current node."""
        return self._layer

    def is_leaf(self):
        """ Whether this node is a leaf node. It is a leaf when the layer called by this node has no sublayers."""
        return isinstance(self._layer,
                          paddle.nn.Layer) and len(self._layer.sublayers()) == 0

    def __hash__(self):
        return self._hash_name

    def __str__(self):
        return self._hash_name


class Graph():
    """ Directed Acyclic Graph used to describe the executing of the model.
    Usually, the graph is built with tracer tools. There is no need to call the constructor directly.
    """

    def __init__(self):
        self._name2node = {}

    def __str__(self):
        return str(self._name2node.keys())

    @property
    def nodes(self) -> List[Node]:
        """ Get all the nodes in the graph.
        """
        return self._name2node.values()

    def find_conv_bn(self) -> List[Tuple[Node, Node]]:
        """ Find all the convolution and batch normalization pairs. 
        Returns:
            A list of convolution and batch normalization pairs. Each pair is a tuple with two
            nodes, the first being the convolution node and the second being the batch normalization node. 
        """
        conv2d_types = [paddle.nn.Conv2D]
        bn_types = [paddle.nn.BatchNorm2D]
        results = []
        for node in self.nodes:
            if type(node.layer) in conv2d_types:
                for _next_node in node.next_nodes:
                    if type(_next_node.layer) in bn_types:
                        results.append((node, _next_node))
                        break
        return results
