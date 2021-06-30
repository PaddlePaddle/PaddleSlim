#copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle
import paddle.nn as nn

from .basic_loss import DMLLoss
from .basic_loss import DistanceLoss
from .basic_loss import RkdDistance
from .basic_loss import RKdAngle


class DistillationDMLLoss(DMLLoss):
    """
    """

    def __init__(self, model_name_pairs=[], act=None, key=None,
                 name="loss_dml"):
        super().__init__(act=act)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.model_name_pairs = model_name_pairs
        self.name = name

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            loss_dict["{}_{}_{}_{}".format(self.name, pair[0], pair[1],
                                           idx)] = super().forward(out1, out2)
        return loss_dict


class DistillationDistanceLoss(DistanceLoss):
    """
    """

    def __init__(self,
                 mode="l2",
                 model_name_pairs=[],
                 key=None,
                 name="loss_distance",
                 **kargs):
        super().__init__(mode=mode, **kargs)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.model_name_pairs = model_name_pairs
        self.name = name + "_" + mode

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            loss = super().forward(out1, out2)
            loss_dict["{}_{}_{}_{}".format(self.name, pair[0], pair[1],
                                           idx)] = loss
        return loss_dict


class DistillationRKDLoss(nn.Layer):
    def __init__(self, model_name_pairs=[], key=None, name="loss_rkd"):
        super().__init__()
        self.model_name_pairs = model_name_pairs
        self.key = key

        self.rkd_angle_loss_func = RKdAngle()
        self.rkd_dist_func = RkdDistance()
        self.name = name

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            loss_dict["{}_{}_{}_angle_{}".format(self.name, pair[0], pair[
                1], idx)] = self.rkd_angle_loss_func(out1, out2)

            loss_dict["{}_{}_{}_dist_{}".format(self.name, pair[0], pair[
                1], idx)] = self.rkd_dist_func(out1, out2)
        return loss_dict
