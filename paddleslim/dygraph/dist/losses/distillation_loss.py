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
from .basic_loss import ATLoss
from .basic_loss import ChannelATLoss
from .basic_loss import FTLoss
from .basic_loss import CCLoss
from .basic_loss import SPLoss
from .basic_loss import NSTLoss
from .basic_loss import ABLoss
from .basic_loss import VIDLoss

__all__ = [
    "DistillationDMLLoss", "DistillationDistanceLoss", "DistillationRKDLoss",
    "DistillationFeatureLoss"
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
        align_ops(list): list of align operations if align.
        transpose_model(list): list of model name to be transposed if align.
        name(string): loss name.
        kwargs(dict): used to build corresponding loss function.
    """

    def __init__(self,
                 mode="l2",
                 model_name_pairs=[],
                 key=None,
                 align_ops=None,
                 transpose_model=[],
                 name="loss_distance",
                 **kwargs):
        super().__init__(mode=mode, **kwargs)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.model_name_pairs = model_name_pairs
        self.name = name + "_" + mode
        self.align_ops = align_ops
        self.transpose_model = transpose_model

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            if self.align_ops is not None and self.align_ops[idx] is not None:
                if self.transpose_model[idx] == 'student':
                    out1 = self.align_ops[idx](out1)
                else:
                    out2 = self.align_ops[idx](out2)
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


class DistillationFeatureLoss(nn.Layer):
    """
    DistillationFeatureLoss
    Args:
        mode: loss mode
        model_name_pairs(list | tuple): model name pairs to extract submodel output.
        key(string | None): key of the tensor used to calculate loss if the submodel.
        align_ops(list): list of align operations if align.
        transpose_model(list): list of model name to be transposed if align.
        name(string): loss name.
    """

    def __init__(self,
                 mode="att",
                 model_name_pairs=[],
                 key=None,
                 align_ops=None,
                 transpose_model=[],
                 name="loss_feature",
                 **kwargs):
        super().__init__()
        assert mode in [
            "att", "channel_att", "ft", "cc", "pkt", "sp", "nst", "ab", "vid"
        ]
        self.key = key
        self.model_name_pairs = model_name_pairs
        self.align_ops = align_ops
        self.transpose_model = transpose_model
        self.name = name
        if mode == 'att':
            self.distill_func = ATLoss(**kwargs)
        elif mode == 'channel_att':
            self.distill_func = ChannelATLoss(**kwargs)
        elif mode == 'ft':
            self.distill_func = FTLoss(**kwargs)
        elif mode == 'cc':
            self.distill_func = CCLoss()
        elif mode == 'sp':
            self.distill_func = SPLoss(**kwargs)
        elif mode == 'nst':
            self.distill_func = NSTLoss()
        elif mode == 'ab':
            self.distill_func = ABLoss(**kwargs)
        elif mode == 'vid':
            if not ("in_channels" in kwargs and "mid_channels" in kwargs and
                    "out_channels" in kwargs):
                raise AssertionError(
                    "In VID, in_channels, mid_channels and out_channels must be in param, but param just contains {}".
                    format(kwargs.keys()))
            in_channels = kwargs.pop("in_channels")
            mid_channels = kwargs.pop("mid_channels")
            out_channels = kwargs.pop("out_channels")
            if not (len(in_channels) == len(out_channels) and
                    len(in_channels) == len(mid_channels) and
                    len(in_channels) == len(model_name_pairs)):
                raise AssertionError(
                    "In VID, len(in_channels) and len(mid_channels) and len(out_channels) and len(model_name_pairs) should be all same, but len(in_channels) is {}, len(out_channels) is {}, len(model_name_pairs) is {}.".
                    format(
                        len(in_channels),
                        len(mid_channels),
                        len(out_channels), len(model_name_pairs)))
            self.distill_func = []

            for idx in range(len(model_name_pairs)):
                self.distill_func.append(
                    VIDLoss(
                        in_channels=in_channels[idx],
                        mid_channels=mid_channels[idx],
                        out_channels=out_channels[idx],
                        **kwargs))

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            if self.align_ops is not None and self.align_ops[idx] is not None:
                if self.transpose_model[idx] == 'student':
                    out1 = self.align_ops[idx](out1)
                else:
                    out2 = self.align_ops[idx](out2)
            if type(self.distill_func) is list:
                loss = self.distill_func[idx](out1, out2)
            else:
                loss = self.distill_func(out1, out2)
            loss_dict["{}_{}_{}_{}".format(self.name, pair[0], pair[1],
                                           idx)] = loss
        return loss_dict
