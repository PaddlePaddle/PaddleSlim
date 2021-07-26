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

from .basic_loss import L1Loss
from .basic_loss import L2Loss
from .basic_loss import SmoothL1Loss
from .basic_loss import CELoss
from .basic_loss import DMLLoss
from .basic_loss import DistanceLoss
from .basic_loss import RKdAngle, RkdDistance

from .distillation_loss import DistillationDistanceLoss
from .distillation_loss import DistillationDMLLoss
from .distillation_loss import DistillationRKDLoss


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
        for config in loss_config_list:
            assert isinstance(config,
                              dict) and len(config) == 1, "yaml format error"
            name = list(config)[0]
            param = config[name]
            assert "weight" in param, "weight must be in param, but param just contains {}".format(
                param.keys())
            self.loss_weight.append(param.pop("weight"))
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
