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
import paddle.nn.functional as F

from paddle.nn import L1Loss
from paddle.nn import MSELoss as L2Loss
from paddle.nn import SmoothL1Loss

__all__ = [
    "CELoss", "DMLLoss", "DistanceLoss", "RKdAngle", "RkdDistance",
    "SpatialATLoss"
]


class ShapeAlign(nn.Layer):
    """
    Align the feature map between student and teacher.
    Args:
        align_type(str): reshape tensor by which op, choice in ['1x1conv','3x3conv','1x1conv+bn','3x3conv+bn','linear']
        in_channel(int): input channel number
        out_channel(int): output channel number
    """

    def __init__(self, align_type, in_channel, out_channel):
        super(ShapeAlign, self).__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        assert align_type.lower() in [
            '1x1conv', '3x3conv', '1x1conv+bn', '3x3conv+bn', 'linear'
        ], "only support 1x1conv, 3x3conv, 1x1conv+bn, 3x3conv+bn, linear for now"
        if align_type.lower() == '1x1conv':
            self.align_op = paddle.nn.Conv2D(
                in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        elif align_type.lower() == '3x3conv':
            self.align_op = paddle.nn.Conv2D(
                in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        elif align_type.lower() == '1x1conv+bn':
            self.align_op = paddle.nn.Sequential(
                paddle.nn.Conv2D(
                    in_channel, out_channel, kernel_size=1, stride=1,
                    padding=0),
                paddle.nn.BatchNorm2D(out_channel))
        elif align_type.lower() == '3x3conv+bn':
            self.align_op = paddle.nn.Sequential(
                paddle.nn.Conv2D(
                    in_channel, out_channel, kernel_size=3, stride=1,
                    padding=1),
                paddle.nn.BatchNorm2D(out_channel))
        elif align_type.lower() == 'linear':
            self.align_op = paddle.nn.Linear(in_channel, out_channel)

    def forward(self, feat):
        assert feat.shape[
            1] == self._in_channel, "input feature channel number must equal to in_channel"
        out = self.align_op(feat)
        return out


class CELoss(nn.Layer):
    """
    CELoss: cross entropy loss
    Args:
        epsilon(float | None): label smooth epsilon. If it is None or not in range (0, 1),
                                  then label smooth will not be used.
        label_act(string | None): activation function, it works when the label is also the logits
                                  rather than the groundtruth label.
        axis(int): axis used to calculate cross entropy loss.

    """

    def __init__(self, epsilon=None, label_act="softmax", axis=-1):
        super().__init__()
        if epsilon is not None and (epsilon <= 0 or epsilon >= 1):
            epsilon = None
        assert label_act in ["softmax", None]
        if epsilon is not None and (epsilon >= 1 or epsilon <= 0):
            epsilon = None
        self.epsilon = epsilon
        self.label_act = label_act
        self.axis = axis

    def _labelsmoothing(self, target, class_num):
        if target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self.epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
        return soft_target

    def forward(self, x, label):
        assert len(x.shape) == len(label.shape), \
            "x and label shape length should be same but got {} for x.shape and {} for label.shape".format(x.shape, label.shape)
        if self.epsilon is not None:
            class_num = x.shape[-1]
            label = self._labelsmoothing(label, class_num)
            x = -F.log_softmax(x, axis=self.axis)
            loss = paddle.sum(x * label, axis=self.axis)
        else:
            if label.shape[self.axis] == x.shape[self.axis]:
                if self.label_act == "softmax":
                    label = F.softmax(label, axis=self.axis)
                soft_label = True
            else:
                soft_label = False
            loss = F.cross_entropy(
                x, label=label, soft_label=soft_label, axis=self.axis)
        loss = loss.mean()
        return loss


class DMLLoss(nn.Layer):
    """
    DMLLoss
    Args:
        act(string | None): activation function used to activate the input tensor
        axis(int): axis used to build activation function
    """

    def __init__(self, act=None, axis=-1):
        super().__init__()
        if act is not None:
            assert act in ["softmax", "sigmoid"]
        if act == "softmax":
            self.act = nn.Softmax(axis=axis)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, out1, out2):
        if self.act is not None:
            out1 = self.act(out1)
            out2 = self.act(out2)

        log_out1 = paddle.log(out1)
        log_out2 = paddle.log(out2)
        loss = (F.kl_div(
            log_out1, out2, reduction='batchmean') + F.kl_div(
                log_out2, out1, reduction='batchmean')) / 2.0
        return loss


class DistanceLoss(nn.Layer):
    """
    DistanceLoss
    Args:
        mode: loss mode
        kargs(dict): used to build corresponding loss function, for more details, please
                     refer to:
                     L1loss: https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/L1Loss_cn.html#l1loss
                     L2Loss: https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/MSELoss_cn.html#mseloss
                     SmoothL1Loss: https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/SmoothL1Loss_cn.html#smoothl1loss
    """

    def __init__(self, mode="l2", **kargs):
        super().__init__()
        assert mode in ["l1", "l2", "smooth_l1"]
        if mode == "l1":
            self.loss_func = nn.L1Loss(**kargs)
        elif mode == "l2":
            self.loss_func = nn.MSELoss(**kargs)
        elif mode == "smooth_l1":
            self.loss_func = nn.SmoothL1Loss(**kargs)

    def forward(self, x, y):
        return self.loss_func(x, y)


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(axis=1)
    prod = paddle.mm(e, e.t())
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clip(
        min=eps)

    if not squared:
        res = res.sqrt()

    return res


class RKdAngle(nn.Layer):
    """
    RKdAngle loss, see https://arxiv.org/abs/1904.05068
    """

    def __init__(self):
        super().__init__()

    def forward(self, student, teacher):
        # reshape for feature map distillation
        bs = student.shape[0]
        student = student.reshape([bs, -1])
        teacher = teacher.reshape([bs, -1])

        td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
        norm_td = F.normalize(td, p=2, axis=2)
        t_angle = paddle.bmm(norm_td, norm_td.transpose([0, 2, 1])).reshape(
            [-1, 1])

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, axis=2)
        s_angle = paddle.bmm(norm_sd, norm_sd.transpose([0, 2, 1])).reshape(
            [-1, 1])
        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss


class RkdDistance(nn.Layer):
    """
    RkdDistance loss, see https://arxiv.org/abs/1904.05068
    Args:
        eps(float): epsilon for the pdist function
    """

    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, student, teacher):
        bs = student.shape[0]
        student = student.reshape([bs, -1])
        teacher = teacher.reshape([bs, -1])

        t_d = pdist(teacher, squared=False, eps=self.eps)
        mean_td = t_d.mean()
        t_d = t_d / (mean_td + self.eps)

        d = pdist(student, squared=False, eps=self.eps)
        mean_d = d.mean()
        d = d / (mean_d + self.eps)

        loss = F.smooth_l1_loss(d, t_d, reduction="mean")
        return loss
