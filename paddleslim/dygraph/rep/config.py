# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

from typing import Dict, Union

import paddle.nn as nn
from .reper import DiverseBranchBlock, ACBlock, RepVGGBlock, SlimRepBlock

SUPPORT_REP_TYPE_LAYERS = [nn.Conv2D, nn.Linear]

__all__ = [
    "BaseRepConfig", "DBBRepConfig", "ACBRepConfig", "RepVGGConfig",
    "SlimRepConfig"
]


class BaseRepConfig:
    """
    Basic reparameterization configuration class.
    Args:
        type_config(dict): Set the reper by the type of layer. The key of `type_config` 
            should be subclass of `paddle.nn.Layer`. Its priority is lower than `layer_config`.
            Default is: `{nn.Conv2D: ACBlock}`.
        layer_config(dict): Set the reper by layer. It has the highest priority among
            all the setting methods. Such as: `{model.conv1: ACBlock}`. Default is None. 
    """

    def __init__(
            self,
            type_config: Dict={nn.Conv2D: ACBlock},
            layer_config: Dict=None, ):
        self._type_config = self._set_type_config(type_config)
        self._layer_config = self._set_layer_config(layer_config)

    def add_config(
            self,
            type_config: Dict=None,
            layer_config: Dict=None, ):
        self._type_config.update(self._set_type_config(type_config))
        self._layer_config.update(self._set_layer_config(layer_config))

    @property
    def all_config(self):
        return {
            'type_config': self._type_config,
            'layer_config': self._layer_config,
        }

    def _set_type_config(self, type_config):
        _type_config = {}
        if type_config:
            for _layer in type_config:
                assert isinstance(_layer, type) and issubclass(
                    _layer, nn.Layer
                ), "Expect to get subclasses under nn.Layer, but got {}.".format(
                    _layer)
                assert _layer in SUPPORT_REP_TYPE_LAYERS, "Expect to get one of `{}`, but got {}.".format(
                    SUPPORT_REP_TYPE_LAYERS, _layer)
                _type_config[_layer] = type_config[_layer]
        return _type_config

    def _set_layer_config(self, layer_config):
        _layer_config = {}
        if layer_config:
            for _layer in layer_config:
                is_support = False
                for support_type in SUPPORT_REP_TYPE_LAYERS:
                    if isinstance(_layer, support_type):
                        is_support = True
                assert is_support, "Expect layer to get one of `{}`.".format(
                    SUPPORT_REP_LAYERS)
                _layer_config[_layer.full_name()] = layer_config[_layer]
        return _layer_config

    def __str__(self):
        result = ""
        if len(self._type_config) > 0:
            result += f"Type config:\n{self._type_config}\n"
        if len(self._layer_config) > 0:
            result += f"Layer config: \n{self._layer_config}\n"
        return result


class DBBRepConfig(BaseRepConfig):
    """
    DBB reparameterization configuration class.
    Args:
        type_config(dict): Set the reper by the type of layer. The key of `type_config` 
            should be subclass of `paddle.nn.Layer`. Its priority is lower than `layer_config`.
            Default is: `{nn.Conv2D: ACBlock}`.
        layer_config(dict): Set the reper by layer. It has the highest priority among
            all the setting methods. Such as: `{model.conv1: ACBlock}`. Default is None. 
    """

    def __init__(
            self,
            type_config: Dict={nn.Conv2D: DiverseBranchBlock},
            layer_config: Dict=None, ):
        self._type_config = self._set_type_config(type_config)
        self._layer_config = self._set_layer_config(layer_config)


class ACBRepConfig(BaseRepConfig):
    """
    ACBlock reparameterization configuration class.
    Args:
        type_config(dict): Set the reper by the type of layer. The key of `type_config` 
            should be subclass of `paddle.nn.Layer`. Its priority is lower than `layer_config`.
            Default is: `{nn.Conv2D: ACBlock}`.
        layer_config(dict): Set the reper by layer. It has the highest priority among
            all the setting methods. Such as: `{model.conv1: ACBlock}`. Default is None. 
    """

    def __init__(
            self,
            type_config: Dict={nn.Conv2D: ACBlock},
            layer_config: Dict=None, ):
        self._type_config = self._set_type_config(type_config)
        self._layer_config = self._set_layer_config(layer_config)


class RepVGGConfig(BaseRepConfig):
    """
    RepVGG reparameterization configuration class.
    Args:
        type_config(dict): Set the reper by the type of layer. The key of `type_config` 
            should be subclass of `paddle.nn.Layer`. Its priority is lower than `layer_config`.
            Default is: `{nn.Conv2D: ACBlock}`.
        layer_config(dict): Set the reper by layer. It has the highest priority among
            all the setting methods. Such as: `{model.conv1: ACBlock}`. Default is None. 
    """

    def __init__(
            self,
            type_config: Dict={nn.Conv2D: RepVGGBlock},
            layer_config: Dict=None, ):
        self._type_config = self._set_type_config(type_config)
        self._layer_config = self._set_layer_config(layer_config)


class SlimRepConfig(BaseRepConfig):
    """
    SlimRepBlock reparameterization configuration class.
    Args:
        type_config(dict): Set the reper by the type of layer. The key of `type_config` 
            should be subclass of `paddle.nn.Layer`. Its priority is lower than `layer_config`.
            Default is: `{nn.Conv2D: ACBlock}`.
        layer_config(dict): Set the reper by layer. It has the highest priority among
            all the setting methods. Such as: `{model.conv1: ACBlock}`. Default is None. 
    """

    def __init__(
            self,
            type_config: Dict={nn.Conv2D: SlimRepBlock},
            layer_config: Dict=None, ):
        self._type_config = self._set_type_config(type_config)
        self._layer_config = self._set_layer_config(layer_config)
