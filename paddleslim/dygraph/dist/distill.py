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
from . import losses

__all__ = ['Distill', 'AdaptorBase']


class LayerConfig:
    def __init__(self,
                 s_feature_idx,
                 t_feature_idx,
                 feature_type,
                 loss_function,
                 weight=1.0,
                 align=False,
                 align_shape=None):
        self.s_feature_idx = s_feature_idx
        self.t_feature_idx = t_feature_idx
        self.feature_type = feature_type
        if loss_function in ['l1', 'l2', 'smooth_l1']:
            self.loss_function = 'DistillationDistanceLoss'
        elif loss_function in ['dml']:
            self.loss_function = 'DistillationDMLLoss'
        elif loss_function in ['rkl']:
            self.loss_function = 'DistillationRKDLoss'
        elif hasattr(losses, loss_function):
            self.loss_function = loss_function
        else:
            raise NotImplementedError("loss function is not support!!!")
        self.weight = weight
        self.align = align
        self.align_shape = align_shape


class AdaptorBase:
    def __init__(self, model):
        self.model = model
        self.add_tensor = False

    def _get_activation(self, outs, name):
        def get_output_hook(layer, input, output):
            outs[name] = output

        return get_output_hook

    def _add_distill_hook(self, outs, mapping_layers_name, layers_type):
        """
            Get output by layer name.
            outs(dict): save the middle outputs of model according to the name.
            mapping_layers(list): name of middle layers.
            layers_type(list): type of the middle layers to calculate distill loss.
        """

        ### TODO: support DP model
        for idx, (n, m) in enumerate(self.model.named_sublayers()):
            if n in mapping_layers_name:
                midx = mapping_layers_name.index(n)
                m.register_forward_post_hook(
                    self._get_activation(outs, layers_type[midx]))

    def mapping_layers(self):
        raise NotImplementedError("function mapping_layers is not implemented")


class Distill(nn.Layer):
    ### TODO: support list of student model and teacher model
    def __init__(self, distill_configs, student_models, teacher_models,
                 adaptors_S, adaptors_T):
        super(Distill, self).__init__()
        assert student_models.training, "The student model should be eval mode."

        self._distill_configs = distill_configs
        self._student_models = student_models
        self._teacher_models = teacher_models
        self._adaptors_S = adaptors_S(self._student_models)
        self._adaptors_T = adaptors_T(self._teacher_models)

        self.stu_outs_dict, self.tea_outs_dict = self._prepare_outputs()

        self.configs = []
        for c in self._distill_configs:
            self.configs.append(LayerConfig(**c).__dict__)

        self.distill_idx = self._get_distill_idx()

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

        # use self._loss_config_list to create all loss object
        self.distill_loss = losses.CombinedLoss(self._loss_config_list)

    def _prepare_outputs(self):
        """
        Add hook to get the output tensor of target layer.
        Returns:
            stu_outs_dict(dict): the name and tensor for the student model,
                such as {'hidden_0': tensor_0, ..}
            tea_outs_dict(dict): the name and tensor for the teather model,
                such as {'hidden_0': tensor_0, ..}    
        """
        stu_outs_dict = collections.OrderedDict()
        tea_outs_dict = collections.OrderedDict()
        stu_outs_dict = self._prepare_hook(self._adaptors_S, stu_outs_dict)
        tea_outs_dict = self._prepare_hook(self._adaptors_T, tea_outs_dict)
        return stu_outs_dict, tea_outs_dict

    def _prepare_hook(self, adaptors, outs_dict):
        """
        Add hook.
        """
        mapping_layers = adaptors.mapping_layers()
        for layer_type, layer in mapping_layers.items():
            if isinstance(layer, str):
                adaptors._add_distill_hook(outs_dict, [layer], [layer_type])
        return outs_dict

    def _get_distill_idx(self):
        """
        For each feature_type, get the feature index in the student and teacher models.
        Returns:
            distill_idx(dict): the feature index for each feature_type,
                such as {'hidden': [[0, 0], [1, 1]], 'out': [[0, 0]]}
        """
        distill_idx = {}
        for config in self._distill_configs:
            if config['feature_type'] not in distill_idx:
                distill_idx[config['feature_type']] = [[
                    int(config['s_feature_idx']), int(config['t_feature_idx'])
                ]]
            else:
                distill_idx[config['feature_type']].append([
                    int(config['s_feature_idx']), int(config['t_feature_idx'])
                ])
        return distill_idx

    def forward(self, *inputs, **kwargs):
        stu_batch_outs = self._student_models.forward(*inputs, **kwargs)
        tea_batch_outs = self._teacher_models.forward(*inputs, **kwargs)
        if not self._teacher_models.training:
            tea_batch_outs = [i.detach() for i in tea_batch_outs]

        # get all target tensor
        if self._adaptors_S.add_tensor == False:
            self._adaptors_S.add_tensor = True
        if self._adaptors_T.add_tensor == False:
            self._adaptors_T.add_tensor = True
        self.stu_outs_dict = self._get_model_intermediate_output(
            self._adaptors_S, self.stu_outs_dict)
        self.tea_outs_dict = self._get_model_intermediate_output(
            self._adaptors_T, self.tea_outs_dict)

        distill_inputs = self._process_outputs()

        ### batch is None just for now
        distill_outputs = self.distill_loss(distill_inputs, None)
        distill_loss = distill_outputs['loss']

        return stu_batch_outs, tea_batch_outs, distill_loss

    def _get_model_intermediate_output(self, adaptors, outs_dict):
        """
        Use the adaptor get the target tensor.
        Returns:
            outs_dict(dict): the name and tensor for the target model,
                such as {'hidden_0': tensor_0, ..}
        """
        mapping_layers = adaptors.mapping_layers()
        for layer_type, layer in mapping_layers.items():
            if isinstance(layer, str):
                continue
            outs_dict[layer_type] = layer
        return outs_dict

    def _process_outputs(self):
        """
        Process the target tensor to adapt for loss.
        """
        ### TODO: support list of student models and teacher_models
        final_distill_dict = {
            "student": collections.OrderedDict(),
            "teacher": collections.OrderedDict()
        }

        for feature_type, dist_idx in self.distill_idx.items():
            for idx, idx_list in enumerate(dist_idx):
                sidx, tidx = idx_list[0], idx_list[1]
                stu_out = self.stu_outs_dict[feature_type + '_' + str(sidx)]
                tea_out = self.tea_outs_dict[feature_type + '_' + str(tidx)]
                if not self._student_models.training:
                    stu_out = stu_out.detach()
                if not self._teacher_models.training:
                    tea_out = tea_out.detach()

                name_str = feature_type + '_' + str(sidx) + '_' + str(tidx)
                final_distill_dict['student'][name_str] = stu_out
                final_distill_dict['teacher'][name_str] = tea_out
        return final_distill_dict
