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
import numpy as np
import collections
from collections import namedtuple
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
            midx = mapping_layers_name.index(n)
            m.register_forward_post_hook(
                _get_activation(outs, layers_type[midx]))


DistillConfig = namedtuple(
    'DistillConfig',
    [
        ### list(dict): config of each mapping layers.
        'layers_config',
    ])
DistillConfig.__new__.__defaults__ = (None, ) * len(DistillConfig._fields)


def transpose_config(config):
    assert 's_feature_idx' in config
    assert 't_feature_idx' in config
    assert 'feature_type' in config
    assert 'loss_function' in config

    if config['loss_function'] in ['l1', 'l2', 'smooth_l1']:
        config['loss_function'] = 'DistillationDistanceLoss'
    elif config['loss_function'] in ['dml']:
        config['loss_function'] = 'DistillationDMLLoss'
    elif config['loss_function'] in ['rkl']:
        config['loss_function'] = 'DistillationRKDLoss'
    else:
        raise NotImplementedError("loss function is not support!!!")
    config['weight'] = config['weight'] if 'weight' in config else 1
    ### TODO: add align in loss
    config['align'] = config['align'] if 'align' in config else False
    return config


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

        self.configs = []
        for c in self._distill_configs:
            self.configs.append(transpose_config(c))

        self.get_distill_idx()
        self._loss_config_list = []
        for c in self.configs:
            loss_config = {}
            loss_config[str(c['loss_function'])] = {}
            loss_config[str(c['loss_function'])]['weight'] = c['weight']
            loss_config[str(c['loss_function'])]['key'] = c[
                'feature_type'] + '_' + str(c['s_feature_idx']) + '_' + str(c[
                    't_feature_idx'])
            ### TODO: support list of student models and teacher_models
            loss_config[str(c['loss_function'])][
                'model_name_pairs'] = [['student', 'teacher']]
            self._loss_config_list.append(loss_config)
        self.prepare_loss()

    def get_distill_idx(self):
        self.distill_idx = {}
        for config in self._distill_configs:
            if config['feature_type'] not in self.distill_idx:
                self.distill_idx[config['feature_type']] = [[
                    int(config['s_feature_idx']), int(config['t_feature_idx'])
                ]]
            else:
                self.distill_idx[config['feature_type']].append([
                    int(config['s_feature_idx']), int(config['t_feature_idx'])
                ])

    def prepare_loss(self):
        self.distill_loss = CombinedLoss(self._loss_config_list)

    def prepare_outputs(self):
        stu_outs_dict = collections.OrderedDict()
        tea_outs_dict = collections.OrderedDict()
        stu_outs_dict = self._adaptors_S(self._student_models, stu_outs_dict)
        tea_outs_dict = self._adaptors_T(self._teacher_models, tea_outs_dict)
        return stu_outs_dict, tea_outs_dict

    def post_outputs(self):
        final_keys = []
        for key, value in self.stu_outs_dict.items():
            if len(key.split('_')) == 1:
                final_keys.append(key)

        ### TODO: support list of student models and teacher_models
        final_distill_dict = {
            "student": collections.OrderedDict(),
            "teacher": collections.OrderedDict()
        }

        for feature_type, dist_idx in self.distill_idx.items():
            for idx, idx_list in enumerate(dist_idx):
                sidx, tidx = idx_list[0], idx_list[1]
                final_distill_dict['student'][feature_type + '_' + str(
                    sidx) + '_' + str(tidx)] = self.stu_outs_dict[
                        feature_type + '_' + str(sidx)]
                final_distill_dict['teacher'][feature_type + '_' + str(
                    sidx) + '_' + str(tidx)] = self.tea_outs_dict[
                        feature_type + '_' + str(tidx)]
        return final_distill_dict

    def forward(self, *inputs, **kwargs):
        stu_batch_outs = self._student_models.forward(*inputs, **kwargs)
        tea_batch_outs = self._teacher_models.forward(*inputs, **kwargs)
        distill_inputs = self.post_outputs()
        ### batch is None just for now
        distill_outputs = self.distill_loss(distill_inputs, None)
        distill_loss = distill_outputs['loss']
        return stu_batch_outs, tea_batch_outs, distill_loss
