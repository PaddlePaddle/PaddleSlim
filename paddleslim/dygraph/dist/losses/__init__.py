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

import copy
import paddle
import paddle.nn as nn

from . import basic_loss
from . import distillation_loss

from .basic_loss import L1Loss
from .basic_loss import L2Loss
from .basic_loss import SmoothL1Loss
from .basic_loss import CELoss
from .basic_loss import DMLLoss
from .basic_loss import DistanceLoss
from .basic_loss import RKdAngle, RkdDistance
from .basic_loss import ShapeAlign
from .basic_loss import SpatialATLoss

from .distillation_loss import DistillationDistanceLoss
from .distillation_loss import DistillationDMLLoss
from .distillation_loss import DistillationRKDLoss
from .distillation_loss import DistillationSpatialATLoss


class CombinedLoss(nn.Layer):
    """
    CombinedLoss: a combination of loss function.
    Args:
        loss_config_list: a config list used to build loss function. A demo is as follows,
                          which is used to calculate dml loss between Student output and
                          Teacher output. Parameter weight is needed for the loss weight.
                            - DistillationDMLLoss:
                                weight: 1.0
                                act: "softmax"
                                model_name_pairs:
                                - ["Student", "Teacher"]
    """

    def __init__(self, loss_config_list=None):
        super().__init__()
        loss_config_list = copy.deepcopy(loss_config_list)
        self.loss_func = []
        self.loss_weight = []
        assert isinstance(loss_config_list, list), (
            'operator config should be a list')
        supported_loss_list = basic_loss.__all__ + distillation_loss.__all__
        for config in loss_config_list:
            assert isinstance(config,
                              dict) and len(config) == 1, "yaml format error"
            name = list(config)[0]
            assert name in supported_loss_list, \
                "loss name must be in {} but got: {}".format(name, supported_loss_list)
            param = config[name]
            assert "weight" in param, "weight must be in param, but param just contains {}".format(
                param.keys())
            self.loss_weight.append(param.pop("weight"))

            # create align ops
            if "align" in param and param.pop("align") == True:
                assert "transpose_model" in param, "To align, transpose_model must be in param, but param just contains {}".format(
                    param.keys())
                assert "align_type" in param, "To align, align_type must be in param, but param just contains {}".format(
                    param.keys())
                assert "in_channels" in param, "To align, in_channels must be in param, but param just contains {}".format(
                    param.keys())
                assert "out_channels" in param, "To align, out_channels must be in param, but param just contains {}".format(
                    param.keys())
                transpose_model = param["transpose_model"]
                align_type = param.pop("align_type")
                in_channels = param.pop("in_channels")
                out_channels = param.pop("out_channels")
                assert type(
                    transpose_model
                ), "To align, transpose_model must be a list, but it is {}".format(
                    type(transpose_model))
                assert type(
                    align_type
                ), "To align, align_type must be a list, but it is {}".format(
                    type(align_type))
                assert type(
                    in_channels
                ), "To align, in_channels must be a list, but it is {}".format(
                    type(in_channels))
                assert type(
                    out_channels
                ), "To align, out_channels must be a list, but it is {}".format(
                    type(out_channels))
                if not (len(in_channels) == len(out_channels) and
                        len(in_channels) == len(param["model_name_pairs"]) and
                        len(in_channels) == len(transpose_model)):
                    raise AssertionError(
                        "To align, len(in_channels) and len(out_channels) and len(model_name_pairs) and len(transpose_model) should be all same, but len(in_channels) is {}, len(out_channels) is {}, len(model_name_pairs) is {},  len(transpose_model) is {}.".
                        format(
                            len(in_channels),
                            len(out_channels),
                            len(param["model_name_pairs"]),
                            len(transpose_model)))
                align_ops = []
                for idx in range(len(param["model_name_pairs"])):
                    if align_type[idx] is not None:
                        align_ops.append(
                            ShapeAlign(align_type[idx], in_channels[idx],
                                       out_channels[idx]))
                    else:
                        align_ops.append(None)

                param['align_ops'] = align_ops

            self.loss_func.append(eval(name)(**param))

    def forward(self, input, batch, **kargs):
        loss_dict = {}
        for idx, loss_func in enumerate(self.loss_func):
            loss = loss_func(input, batch, **kargs)
            weight = self.loss_weight[idx]
            if isinstance(loss, paddle.Tensor):
                loss = {"loss_{}_{}".format(str(loss), idx): loss * weight}
            else:
                loss = {
                    "{}_{}".format(key, idx): loss[key] * weight
                    for key in loss
                }
            loss_dict.update(loss)
        loss_dict["loss"] = paddle.add_n(list(loss_dict.values()))
        return loss_dict
