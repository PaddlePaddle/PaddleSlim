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

import numpy as np
import paddle

from .basic_loss import BASIC_LOSS

__all__ = ["DistillationLoss", "ShapeAlign"]


class DistillationLoss(paddle.nn.Layer):
    """
    DistillationLoss
    Args:
        model_name_pairs(list | tuple): model name pairs to extract submodel output.
        layers_name(list(string)): keys of the tensor used to calculate loss if the submodel.
        loss_function(string): the name of loss function.
        temperature(float): the temperature to compute distill loss.
    """

    def __init__(self,
                 model_name_pairs=[],
                 layers_name=None,
                 loss_function=None,
                 temperature=1.0,
                 **params):
        super().__init__()
        self.model_name_pairs = model_name_pairs
        self.layers_name = layers_name
        self.loss_function = loss_function
        self.temperature = temperature
        self.align_params = params.pop(
            'align_params') if 'align_params' in params else None
        if self.align_params is not None:
            if 'transpose_model' in self.align_params:
                self.transpose_model = self.align_params['transpose_model']
                self.align_params.pop('transpose_model')
            else:
                self.transpose_model = 'student'
            self.align_func = ShapeAlign(**self.align_params)

        self.loss_func = BASIC_LOSS.get(loss_function)(**params)

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.layers_name != None:
                assert len(self.layers_name
                           ) == 2, "length of layers_name must be equal to 2."
                out1 = out1[self.layers_name[0]]
                out2 = out2[self.layers_name[1]]
            if self.align_params is not None:
                if self.transpose_model == 'student':
                    out1 = self.align_func(out1)
                else:
                    out2 = self.align_func(out2)
            if self.temperature != 1.0:
                out1 = out1 / self.temperature
                out2 = out2 / self.temperature
            loss_dict["{}_{}_{}_{}_{}".format(self.loss_function, pair[0], pair[
                1], self.layers_name[0] if self.layers_name != None else "0", \
                self.layers_name[1] if self.layers_name != None else "0")] = self.loss_func(out1, out2)
        return loss_dict


class ShapeAlign(paddle.nn.Layer):
    """
    Align the feature map between student and teacher.
    Args:
        align_type(str): reshape tensor by which op, choice in ['1x1conv','3x3conv','1x1conv+bn','3x3conv+bn','linear']
        in_channel(int): input channel number
        out_channel(int): output channel number
    """

    def __init__(self, align_type, in_channel, out_channel, weight_init=None):
        super(ShapeAlign, self).__init__()
        self._in_channel = in_channel
        self._out_channel = out_channel
        assert align_type.lower() in [
            '1x1conv', '3x3conv', '1x1conv+bn', '3x3conv+bn', 'linear'
        ], "only support 1x1conv, 3x3conv, 1x1conv+bn, 3x3conv+bn, linear for now"

        bias_attr = None
        if weight_init is not None:
            assert 'initializer' in weight_init
            init_mode = weight_init.pop('initializer')
            ### load transpose weight from pretrained model.
            if init_mode == 'Assign':
                bias = None
                assert 'params_path' in weight_init
                assert 'params_name' in weight_init
                params_path = weight_init['params_path']
                params_name = weight_init['params_name']
                if isinstance(weight_init['params_name'], (list, tuple)):
                    assert len(weight_init['params_name']) <= 2
                    weight = paddle.load(params_path)[params_name[0]]
                    bias = paddle.load(params_path)[params_name[1]]
                else:
                    weight = paddle.load(params_path)[params_name]
                weight_attr = paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.Assign(weight))
                if bias is not None:
                    bias_attr = paddle.framework.ParamAttr(
                        initializer=paddle.nn.initializer.Assign(bias))
            else:
                weight_attr = paddle.framework.ParamAttr(initializer=eval(
                    'paddle.nn.initializer.{}'.format(init_mode))(
                        **weight_init))
        else:
            weight_attr = None
        if align_type.lower() == '1x1conv':
            self.align_op = paddle.nn.Conv2D(
                in_channel,
                out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                weight_attr=weight_attr,
                bias_attr=bias_attr)
        elif align_type.lower() == '3x3conv':
            self.align_op = paddle.nn.Conv2D(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=weight_attr,
                bias_attr=bias_attr)
        elif align_type.lower() == '1x1conv+bn':
            self.align_op = paddle.nn.Sequential(
                paddle.nn.Conv2D(
                    in_channel,
                    out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    weight_attr=weight_attr,
                    bias_attr=bias_attr),
                paddle.nn.BatchNorm2D(out_channel))
        elif align_type.lower() == '3x3conv+bn':
            self.align_op = paddle.nn.Sequential(
                paddle.nn.Conv2D(
                    in_channel,
                    out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    weight_attr=weight_attr,
                    bias_attr=bias_attr),
                paddle.nn.BatchNorm2D(out_channel))
        elif align_type.lower() == 'linear':
            self.align_op = paddle.nn.Linear(
                in_channel,
                out_channel,
                weight_attr=weight_attr,
                bias_attr=bias_attr)

    def forward(self, feat):
        if isinstance(feat, tuple):
            feat = feat[0]
        out = self.align_op(feat)
        return out
