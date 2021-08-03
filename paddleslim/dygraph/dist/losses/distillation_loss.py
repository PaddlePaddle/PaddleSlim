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
from .basic_loss import ShapeAlign

__all__ = [
    "DistillationDMLLoss",
    "DistillationDistanceLoss",
    "DistillationRKDLoss",
]


class DistillationDMLLoss(DMLLoss):
    """
    DistillationDMLLoss
    Args:
        model_name_pairs(list | tuple): model name pairs to extract submodel output.
        act(string | None): activation function used to build dml loss.
        axis(int): axis used to build activation function.
        key(string | None): key of the tensor used to calculate loss if the submodel
                            output type is dict.
        name(string): loss name.
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
    DistillationDistanceLoss
    Args:
        mode: loss mode
        model_name_pairs(list | tuple): model name pairs to extract submodel output.
        key(string | None): key of the tensor used to calculate loss if the submodel.
        name(string): loss name.
        kargs(dict): used to build corresponding loss function.
    """

    def __init__(self,
                 mode="l2",
                 model_name_pairs=[],
                 key=None,
                 align=False,
                 transpose_model=None,
                 align_type=[],
                 in_channels=[],
                 out_channels=[],
                 name="loss_distance",
                 **kargs):
        super().__init__(mode=mode, **kargs)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.model_name_pairs = model_name_pairs
        self.name = name + "_" + mode
        if align:
            assert transpose_model is not None, "if align is True, please set transpose_model"
            assert transpose_model in [
                'teacher', 'student'
            ], "please choose 'teacher' or 'student' to be transposed"
            assert align_type is not None, "if align is True, please set align_type"
            assert len(
                in_channels) > 0, "if align is True, please set in_channels"
            assert len(
                out_channels) > 0, "if align is True, please set out_channels"

        self.transpose_model = transpose_model
        self.align_type = align_type

        self.align = None
        if align:
            self.align = nn.LayerList()
            for idx in range(len(model_name_pairs)):
                if align_type[idx] is not None:
                    name = '{}'.format(idx)
                    self.align.add_sublayer(name,
                                            ShapeAlign(align_type[idx],
                                                       in_channels[idx],
                                                       out_channels[idx]))

    def forward(self, predicts, batch):
        loss_dict = dict()
        align_idx = 0
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]

            if self.align is not None and self.align_type[idx] is not None:
                if self.transpose_model == 'student':
                    out1 = self.align[align_idx](out1)
                else:
                    out2 = self.align[align_idx](out2)
                align_idx += 1
            loss = super().forward(out1, out2)
            loss_dict["{}_{}_{}_{}".format(self.name, pair[0], pair[1],
                                           idx)] = loss
        return loss_dict


class DistillationRKDLoss(nn.Layer):
    """
    DistillationRKDLoss
    Args:
        model_name_pairs(list | tuple): model name pairs to extract submodel output.
        key(string | None): key of the tensor used to calculate loss if the submodel.
        eps(float): epsilon for the pdist function for RkdDistance loss.
        name(string): loss name.
    """

    def __init__(self,
                 model_name_pairs=[],
                 key=None,
                 eps=1e-12,
                 name="loss_rkd"):
        super().__init__()
        self.model_name_pairs = model_name_pairs
        self.key = key

        self.rkd_angle_loss_func = RKdAngle()
        self.rkd_dist_func = RkdDistance(eps=eps)
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
