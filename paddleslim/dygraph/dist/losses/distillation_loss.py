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
from .basic_loss import KLLoss

__all__ = [
    "DistillationDMLLoss",
    "DistillationDistanceLoss",
    "DistillationRKDLoss",
    "SegPairWiseLoss",
    "SegChannelwiseLoss",
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
            such as [['student', 'teacher']]
        key(string | None): key of the tensor used to calculate loss if the submodel.
            such as 'hidden_0_0'
        name(string): loss name.
        kargs(dict): used to build corresponding loss function.
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
            if isinstance(out1, list):
                assert len(out1) == 1
            if isinstance(out2, list):
                assert len(out2) == 1
            loss = super().forward(out1[0], out2[0])
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


class SegPairWiseLoss(DistanceLoss):
    """
    Segmentation pairwise loss, see https://arxiv.org/pdf/1903.04197.pdf

    Args:
        model_name_pairs(list | tuple): model name pairs to extract submodel output.
        key(string): key of the tensor used to calculate loss if the submodel
                            output type is dict.
        mode(string, optional): loss mode. It supports l1, l2 and smooth_l1. Default: l2.
        reduction(string, optional): the reduction params for F.kl_div. Default: mean.
        name(string, optional): loss name. Default: seg_pair_wise_loss.
    """

    def __init__(self,
                 model_name_pairs=[],
                 key=None,
                 mode="l2",
                 reduction="mean",
                 name="seg_pair_wise_loss"):
        super().__init__(mode=mode, reduction=reduction)

        assert isinstance(model_name_pairs, list)
        assert key is not None
        self.key = key
        self.model_name_pairs = model_name_pairs
        self.name = name

        self.pool1 = nn.AdaptiveAvgPool2D(output_size=[2, 2])
        self.pool2 = nn.AdaptiveAvgPool2D(output_size=[2, 2])

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]][self.key]
            out2 = predicts[pair[1]][self.key]

            pool1 = self.pool1(out1)
            pool2 = self.pool2(out2)

            loss_name = "{}_{}_{}_{}".format(self.name, pair[0], pair[1], idx)
            loss_dict[loss_name] = super().forward(pool1, pool2)
        return loss_dict


class SegChannelwiseLoss(KLLoss):
    """
    Segmentation channel wise loss, see `Channel-wise Distillation for Semantic Segmentation`.
    Args:
        model_name_pairs(list | tuple): model name pairs to extract submodel output.
        key(string): key of the tensor used to calculate loss if the submodel
                            output type is dict.
        act(string, optional): activation function used for the input and label tensor.
            Default: softmax.
        axis(int, optional): the axis for the act. Default: -1.
        reduction(str, optional): the reduction params for F.kl_div. Default: mean.
        name(string, optional): loss name. Default: seg_ch_wise_loss.
    """

    def __init__(self,
                 model_name_pairs=[],
                 key=None,
                 act='softmax',
                 axis=-1,
                 reduction="mean",
                 name="seg_ch_wise_loss"):
        super().__init__(act, axis, reduction)

        assert isinstance(model_name_pairs, list)
        assert key is not None
        self.model_name_pairs = model_name_pairs
        self.key = key
        self.name = name

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]][self.key]
            out2 = predicts[pair[1]][self.key]
            loss_name = "{}_{}_{}_{}".format(self.name, pair[0], pair[1], idx)
            loss_dict[loss_name] = super().forward(out1, out2)
        return loss_dict
