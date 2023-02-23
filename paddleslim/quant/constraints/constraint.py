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

__all__ = ["FusionConstraint", "Constraint"]


class Constraint():
    """在量化训练或离线量化过程中，需要遵循的约束。可以是且不限于以下几种约束：
    1. Operators Fusion: 将多个Operators当做一个融合的Operator，只量化融合Operator的输入和输出
    2. 多个Tensors的量化相互影响：比如多个Tensors量化参数需要保持一致，或需要满足更复杂的要求
    3. 统计量化参数的过程与常规的forward流程不一样，需要特殊处理
    """

    def apply(self, model, graph, qconfig):
        """将约束应用到目标模型上，并更新量化配置信息。应该在量化训练和离线量化校准操作前执行该方法。
        该方法会直接inplace地对model和qconfig进行操作。
        该方法为抽象方法，所有继承Constraint的子类都应该实现该方法。
        """
        pass


class FusionConstraint(Constraint):
    """Define some functoins used to fuse operators.
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

    def fuse_ops(self, model, fused_layer_type, layers, config):
        """将模型中多层融合为一层。
        """
        fused_layer = fused_layer_type(*layers)

        for _layer in layers[1:]:
            self._replace_layer(model, _layer, paddle.nn.Identity())
            if _layer in config._layer2config:
                config._layer2config.pop(_layer)

        self._replace_layer(model, layers[0], fused_layer)
        config._layer2config[fused_layer] = config._layer2config[layers[0]]
