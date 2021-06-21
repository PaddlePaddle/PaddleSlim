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
import os
from collections import namedtuple
import paddle
import paddle.nn as nn
from .losses import *

__all__ = ['DistillConfig', 'Distill', 'add_distill_hook']


def _get_activation(outs, name):
    def get_output_hook(layer, input, output):
        outs[name] = output

    return get_output_hook


def add_distill_hook(net, outs, mapping_layers_name, layers_type):
    """
        Get output by name.
        outs(dict): save the middle outputs of model according to the name.
        mapping_layers(list): name of middle layers.
        layers_type(list): type of the middle layers to calculate distill loss.
    """
    ### TODO: support DP model
    for idx, (n, m) in enumerate(net.named_sublayers()):
        if n in mapping_layers_name:
            print(n, mapping_layers_name)
            midx = mapping_layers_name.index(n)
            m.register_forward_post_hook(
                _get_activation(outs, layers_type[midx] + '_' + str(idx)))


DistillConfig = namedtuple(
    'DistillConfig',
    [
        ### list(dict): config of each mapping layers.
        'layers_config',
    ])
DistillConfig.__new__.__defaults__ = (None, ) * len(DistillConfig._fields)


class Config:
    def __init__(self,
                 layer_S,
                 layer_T,
                 feature_type,
                 loss_function,
                 weight=1.0,
                 align=None):
        self.layer_S = layer_S
        self.layer_T = layer_T
        self.feature_type = feature_type
        if loss_function in ['l1', 'l2', 'smooth_l1']:
            self.loss_function = 'DistillationDistanceLoss'
        elif loss_function in ['dml']:
            self.loss_function = 'DistillationDMLLoss'
        elif loss_function in ['rkl']:
            self.loss_function = 'DistillationRKDLoss'
        else:
            raise NotImplementedError("loss function is not support!!!")
        self.weight = weight
        self.align = align

    @classmethod
    def from_dict(cls, obj):
        if obj is None:
            return None
        else:
            return cls(**obj)


class Distill(nn.Layer):
    ### TODO: support list of student model and teacher model
    def __init__(self, distill_configs, student_models, teacher_models,
                 adaptors_S, adaptors_T):
        super(Distill, self).__init__()
        self._distill_configs = distill_configs
        self._student_models = student_models
        self._teacher_models = teacher_models
        self._adaptors_S = adaptors_S
        self._adaptors_T = adaptors_T

        self.stu_outs_dict, self.tea_outs_dict = self.prepare_outputs()

        self.configs = [Config.from_dict(c) for c in self._distill_configs]
        self._loss_config_list = []
        loss_config = {}
        for c in self.configs:
            loss_config[str(c.loss_function)] = {}
            cd = c.__dict__
            loss_config[str(c.loss_function)]['weight'] = cd['weight']
            loss_config[str(c.loss_function)]['key'] = cd['feature_type']
            loss_config[str(c.loss_function)]['model_name_pairs'] = cd[
                'feature_type']
        self._loss_config_list.append(loss_config)
        self.prepare_loss()

    def prepare_loss(self):
        CombinedLoss(self._loss_config_list)

    def prepare_outputs(self):
        stu_outs = self._adaptors_S(self._student_models)
        tea_outs = self._adaptors_T(self._teacher_models)
        return stu_outs, tea_outs
