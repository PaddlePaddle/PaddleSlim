"""Define structures that store relationship of tensors to be pruned."""
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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

__all__ = ['PruneInfo', 'Group']


class PruneInfo(object):
    """
    The description of one pruning operation.
    Args:
        name(str): The name of tensor to be pruned.
        axis(int): The axis to be pruned on.
        transform(dict): Information used to convert pruned indexes of master
                         tensor to indexes of current tensor.
        op(OpWrapper): The operator with current tensor as input.
        is_parameter(bool): whether the tensor is parameter. Default: True.
    """

    def __init__(self, name, axis, transform, op, is_parameter=True):
        assert (isinstance(name, str),
                "name should be str, but get type = ".format(type(name)))
        assert (isinstance(axis, int))
        self.name = name
        self.axis = axis
        if isinstance(self.axis, int):
            self.axis = [self.axis]
        self.transform = transform
        self.op = op
        self.is_parameter = is_parameter

    def __equal__(self, other):
        if self.name != other.name:
            return False
        if self.axis != other.axis:
            return False
        for _key in self.transform:
            if _key not in other.transform:
                return False
            if self.transform[key] != other.transform[_key]:
                return Flase
        return True


class Group(object):
    """
    A group of pruning operations.
    
      conv1-->conv2-->batch_norm

    For the network defined above, if weight of conv1 is pruned on 0-axis,
    weight of'conv2' should be pruned on 1-axis. The pruning operations on 0-axis of
    'conv1' and that on 1-aixs of 'conv2' is a group. And the {'name': conv1.weight_name, 'axis': 0}
    is the master of current group.
     
    Args:
        master(dict): The master pruning operation.
    """

    def __init__(self, master=None):
        self._master = master
        self._nodes = {}

    def variables(self):
        """
        Get all tensors to be pruned in current group.
        Returns:
          list<str>: Names of tensor to be pruned.
        """
        return list(self._nodes.keys())

    def __equal__(self, other):
        if len(self.nodes) != len(other.nodes):
            return False
        for name in self.nodes:
            if name not in other.nodes:
                return False
            if len(self.nodes[name]) != len(other.nodes[name]):
                for _node in self.nodes[name]:
                    if _node not in other.nodes[name]:
                        return False
        return True

    def extend(self, nodes):
        """
        Extend current group.

        Args:
            nodes(list<PruneInfo>): A list of pruning operations to be added into current group. 
        """
        for _node in nodes:
            self.add(_node)

    def add(self, node):
        """
        Add a pruning operation into current group.
        Args:
            node(PruneInfo): Pruning operation to be added into current group.
        """
        assert (isinstance(node, PruneInfo))
        if self._master is None:
            # the first added pruning operation will be master.
            self._master = {"name": node.name, "axis": node.aixs}
        if node.name not in self._nodes:
            self._nodes[node.name] = []
        if node not in self._nodes[node.name]:
            self._nodes[node.name].append(node)

    @property
    def master(self):
        return self._master

    def get_prune_info(self, tensor_name, axis=None):
        """
        Get pruning operations of target tensor.
        Args:
            tensor_name(str): The name of tensor to be pruned.
            axis(axis): The axis to be pruned on.
        Returns:
            list<PruneInfo>: Pruning operations. None means pruning
                             operations not found.
        """
        if tensor_name not in self._nodes:
            return None
        if axis is None:
            return self._nodes[tensor_name]
        ret = [
            _info for _info in self._nodes[tensor_name] if _info.axis == axis
        ]
        ret = ret if len(ret) != 0 else None
        return ret

    def all_prune_info(self):
        """
        Get all pruning operations in current group.
        Returns:
            list<PruneInfo>: Pruning operations.
        """
        ret = []
        for _items in self._nodes.values():
            ret.extend(_items)
        return ret
