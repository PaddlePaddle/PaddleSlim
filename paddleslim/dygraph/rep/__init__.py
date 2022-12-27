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
from .diversebranchblock import DiverseBranchBlock
from .acblock import ACBlock
_logger = get_logger(__name__, level=logging.INFO)

__all__ = ["Reparameterization"]

SUPPORT_REP_DICT = {"DBB": DiverseBranchBlock, "ACB": ACBlock}


class Reparameterization:
    """
    Re-parameterization interface of dygraph model.
    Args:
        model(nn.Layer): Model of networks.
        algo(str): Reparameterized algorithm, currently supports: `DBB`, `ACB`.
        op_type(str): The layer that the reparameterization module will replace, currently supports `conv_bn`.
    """

    def __init__(self, algo="DBB", op_type="conv_bn"):
        self.model = None
        assert algo in list(SUPPORT_REP_DICT.keys(
        )), "algo currently supports {}, but got {}.".format(
            list(SUPPORT_REP_DICT.keys()), algo)
        assert op_type in [
            "conv_bn"
        ], "op_type currently supports `conv_bn`, but git {}".format(op_type)
        self.reper = SUPPORT_REP_DICT[algo]
        self.op_type = op_type

    def __call__(self, model):
        """
        Re-parameterization callback interface.
        Args:
            model(nn.Layer): The model to be reparameterized.
        """
        if not self.model:
            self.model = model
        if self.op_type == "conv_bn":
            conv_bn_pairs = self._get_conv_bn_pair(self.model)
            if not conv_bn_pairs:
                _logger.info(
                    "No conv-bn layer found, so skip the reparameterization.")
                return
            self._replace_conv_bn_with_rep(self.model, conv_bn_pairs)

    def convert_to_deploy(self):
        """
        Re-parameterization export interface.
        Args:
            model(nn.Layer): The model that has been reparameterized.
        """
        for layer in self.model.sublayers():
            if hasattr(layer, 'convert_to_deploy'):
                layer.convert_to_deploy()

    def _get_conv_bn_pair(self, model):
        conv_bn_pairs = []
        tmp_pair = [None, None]
        for name, layer in model.named_sublayers():
            if isinstance(layer, nn.Conv2D):
                tmp_pair[0] = name
            if isinstance(layer, nn.BatchNorm2D) or isinstance(layer,
                                                               nn.BatchNorm):
                tmp_pair[1] = name

            if tmp_pair[0] and tmp_pair[1] and len(tmp_pair) == 2:
                conv_bn_pairs.append(tmp_pair)
                tmp_pair = [None, None]
        return conv_bn_pairs

    def _replace_conv_bn_with_rep(self, model, conv_bn_pairs):
        for conv_bn_pair in conv_bn_pairs:
            for layer_name in conv_bn_pair:
                parent_layer, sub_name = self._find_parent_layer_and_sub_name(
                    model, layer_name)
                module = getattr(parent_layer, sub_name)
                if isinstance(module, nn.Conv2D):
                    new_layer = self.reper(
                        in_channels=module._in_channels,
                        out_channels=module._out_channels,
                        kernel_size=module._kernel_size[0],
                        stride=module._stride,
                        groups=module._groups)
                    setattr(parent_layer, sub_name, new_layer)

                if isinstance(module, nn.BatchNorm2D) or isinstance(
                        module, nn.BatchNorm):
                    new_layer = nn.Identity()
                    setattr(parent_layer, sub_name, new_layer)
        return model

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
