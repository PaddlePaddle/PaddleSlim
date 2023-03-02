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

import paddle
import abc
from typing import List
from paddleslim.core.graph import Graph
from paddleslim.quant.config import SlimQuantConfig

__all__ = ["FusionConstraint", "Constraint"]


class Constraint(metaclass=abc.ABCMeta):
    """ Constraints are the rules applied on layers during quantization-aware training.
    The abstract class defined the interface of constraint. All the constraints should
    extend this abstract class and implement the 'apply' function.
    """

    @abc.abstractmethod
    def apply(self,
              model: paddle.nn.Layer,
              graph: Graph,
              qconfig: SlimQuantConfig) -> None:
        """ Apply the rules of the constraint on the model in place.
        Usually, there are three steps to implement the apply function:
        1. Find all the targets to be constrained. Each target can be one layer or a group of layers.
        2. If the target is a group of layers, fuse it into a fusion layer.
        3. Mapping the fusion layer to a QAT strategy layer by the "add_qat_layer_mapping" function of "qconfig".
        
        Args:
        - model(paddle.nn.Layer): The model to be constrained.
        - graph(Graph): A structure stored the DAG of executing model. It is used to
        help find layers by some pattern.
        - qconfig(SlimQuantConfig): The configuration of quantization.
        """
        pass


class FusionConstraint(Constraint, metaclass=abc.ABCMeta):
    """ Define some functions used to fuse operators.
    """

    def _find_parent_and_sublayer_name(self, model, layer):
        for _name, _sub_layer in model.named_sublayers():
            if layer.full_name() == _sub_layer.full_name():
                return model, _name
            else:
                result = self._find_parent_and_sublayer_name(_sub_layer, layer)
                if result is not None:
                    return result

    def _replace_layer(self, model, source, target):
        result = self._find_parent_and_sublayer_name(model, source)
        assert result is not None, f"Can't find the parent layer of {source.full_name()} in model {model.full_name()}!!!"
        parent_layer, sub_name = result
        parent_layer._sub_layers[sub_name] = target
        setattr(parent_layer, sub_name, target)

    def fuse_ops(self,
                 model: paddle.nn.Layer,
                 fused_layer_type,
                 layers: List[paddle.nn.Layer],
                 config: SlimQuantConfig):
        """ Fuse layers into fusion layer.
        Args:
            - model(paddle.nn.Layer): The model whose sublayers will be fused. 
            - fused_layer_type(type): Type of fusion layer.
            - layers(list): List of layers to be fused. It will be the arguments to create 'fused_layer_type' instance
            by calling fused_layer_type(*layers).
            - configs(SlimQuantConfig): The configuration of quantization.
        """
        fused_layer = fused_layer_type(*layers)

        for _layer in layers[1:]:
            self._replace_layer(model, _layer, paddle.nn.Identity())
            if _layer in config._layer2config:
                config._layer2config.pop(_layer)

        self._replace_layer(model, layers[0], fused_layer)
        config._layer2config[fused_layer] = config._layer2config[layers[0]]
