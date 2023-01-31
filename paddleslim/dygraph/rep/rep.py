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

import logging
import paddle.nn as nn

from ...common import get_logger
from .config import BaseRepConfig, SUPPORT_REP_TYPE_LAYERS

_logger = get_logger(__name__, level=logging.INFO)

__all__ = ["Reparameter"]


class Reparameter:
    """
    Re-parameterization interface of dygraph model.
    Args:
        model(nn.Layer): Model of networks.
        config(instance): Reparameterization config, default is `BaseRepConfig`.
    """

    def __init__(self, config=BaseRepConfig):
        assert config != None, "config cannot be None."
        self._config = config.all_config
        self._layer2reper_config = {}

    def prepare(self, model):
        """
        Re-parameterization prepare model callback interface.
        Args:
            model(nn.Layer): The model to be reparameterized.
        """
        self._layer2reper_config = self._parser_rep_config(model)
        # Conv2D
        if "Conv2D" in self._layer2reper_config:
            conv2d2reper_config = self._layer2reper_config["Conv2D"]
            conv_bn_pairs = self._get_conv_bn_pair(model)
            if not conv_bn_pairs:
                _logger.info(
                    "No conv-bn layer found, so skip the reparameterization.")
                return model
            for layer_name in conv2d2reper_config:
                if layer_name in list(conv_bn_pairs.keys()):
                    per_conv_bn_pair = [layer_name, conv_bn_pairs[layer_name]]
                    self._replace_conv_bn_with_reper(
                        model, conv2d2reper_config[layer_name],
                        per_conv_bn_pair)
        return model

    def convert(self, model):
        """
        Re-parameterization export interface, it will run fusion operation.
        Args:
            model(nn.Layer): The model that has been reparameterized.
        """
        for layer in model.sublayers():
            if hasattr(layer, 'convert_to_deploy'):
                layer.convert_to_deploy()

    def _parser_rep_config(self, model):
        _layer2reper_config = {}
        for name, layer in model.named_sublayers():
            support_type_layers = list(self._config['type_config'].keys())
            refine_layer_full_names = list(self._config['layer_config'].keys())
            cur_layer_reper = None
            # Firstly, parser type layer in model.
            for layer_type in support_type_layers:
                if isinstance(layer, layer_type):
                    cur_layer_reper = self._config['type_config'][layer_type]

            # Secondly, parser layer full name in model.
            if name in refine_layer_full_names:
                cur_layer_reper = self._config['layer_config'][name]

            # Conv2d
            if cur_layer_reper and isinstance(layer, nn.Conv2D):
                if "Conv2D" in _layer2reper_config:
                    _layer2reper_config["Conv2D"].update({
                        name: cur_layer_reper
                    })
                else:
                    _layer2reper_config["Conv2D"] = {name: cur_layer_reper}
            # Linear
            elif cur_layer_reper and isinstance(layer, nn.Linear):
                if "Linear" in _layer2reper_config:
                    _layer2reper_config["Linear"].update({
                        name: cur_layer_reper
                    })
                else:
                    _layer2reper_config["Linear"] = {name: cur_layer_reper}
            elif cur_layer_reper:
                _logger.info(
                    "{} not support reparameterization, please choose one of {}".
                    format(name, SUPPORT_REP_TYPE_LAYERS))
        return _layer2reper_config

    def _get_conv_bn_pair(self, model):
        """
        Get the combination of Conv2D and BatchNorm2D.
        Args:
            model(nn.Layer): The model that has been reparameterized.
        """
        conv_bn_pairs = {}
        tmp_pair = [None, None]
        for name, layer in model.named_sublayers():
            if isinstance(layer, nn.Conv2D):
                tmp_pair[0] = name
            if isinstance(layer, nn.BatchNorm2D) or isinstance(
                    layer, nn.BatchNorm):
                tmp_pair[1] = name

            if tmp_pair[0] and tmp_pair[1] and len(tmp_pair) == 2:
                conv_bn_pairs[tmp_pair[0]] = tmp_pair[1]
                tmp_pair = [None, None]
        return conv_bn_pairs

    def _replace_conv_bn_with_reper(self, model, reper, conv_bn_pair):
        """
        Replace Conv2D and BatchNorm2D with reper.
        Args:
            model(nn.Layer): The model that has been reparameterized.
            reper(nn.Layer): The reper used by the current layer.
            conv_bn_pairs(list[str, str]): List of combination of Conv2D and BatchNorm2D.
        """
        for layer_name in conv_bn_pair:
            parent_layer, sub_name = self._find_parent_layer_and_sub_name(
                model, layer_name)
            module = getattr(parent_layer, sub_name)
            if isinstance(module, nn.Conv2D):
                new_layer = reper(
                    in_channels=module._in_channels,
                    out_channels=module._out_channels,
                    kernel_size=module._kernel_size[0],
                    stride=module._stride[0],
                    groups=module._groups,
                    padding=module._padding)
                setattr(parent_layer, sub_name, new_layer)

            if isinstance(module, nn.BatchNorm2D) or isinstance(
                    module, nn.BatchNorm):
                new_layer = nn.Identity()
                setattr(parent_layer, sub_name, new_layer)

    def _find_parent_layer_and_sub_name(self, model, name):
        """
        Given the model and the name of a layer, find the parent layer and
        the sub_name of the layer.
        For example, if name is 'block_1/convbn_1/conv_1', the parent layer is
        'block_1/convbn_1' and the sub_name is `conv_1`.
        Args:
            model(paddle.nn.Layer): the model to be reparameterized.
            name(string): the name of a layer.

        Returns:
            parent_layer, subname
        """
        assert isinstance(model, nn.Layer), \
                "The model must be the instance of paddle.nn.Layer."
        assert len(name) > 0, "The input (name) should not be empty."

        last_idx = 0
        idx = 0
        parent_layer = model
        while idx < len(name):
            if name[idx] == '.':
                sub_name = name[last_idx:idx]
                if hasattr(parent_layer, sub_name):
                    parent_layer = getattr(parent_layer, sub_name)
                    last_idx = idx + 1
            idx += 1
        sub_name = name[last_idx:idx]
        return parent_layer, sub_name
