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
import copy
import collections
import numpy as np
import paddle
import paddle.nn as nn
from . import losses
from .losses.basic_loss import BASIC_LOSS
from .distill_helpers import yaml2config

__all__ = ['Distill']


class LayerConfig:
    """ The key of config can be set"""

    def __init__(self,
                 model_name_pairs,
                 layers_name,
                 loss_function,
                 io=["output", "output"],
                 idx=[None, None],
                 weight=1.0,
                 temperature=1.0,
                 align_params=None,
                 **loss_params):
        self.model_name_pairs = model_name_pairs
        self.layers_name = layers_name
        if loss_function not in BASIC_LOSS.module_dict:
            raise NotImplementedError("loss function {} is not support. "
                                      "Support loss including {}".format(
                                          loss_function,
                                          BASIC_LOSS.module_dict.keys()))
        self.loss_function = loss_function
        self.io = io
        self.idx = idx
        self.weight = weight
        self.temperature = temperature
        self.align_params = align_params
        for k, v in loss_params.items():
            setattr(self, k, v)


def _add_hooks(model, outs, layers_name, hook_layers_name, io='o', idx="None"):
    """
        Get output by layer name.
        models(nn.Layer):  model need to be add hook.
        outs(dict): save the middle outputs of model according to the name.
        hook_layers_name(list): name of middle layers.
    """

    def _get_activation(outs, name, io, idx):
        def get_output_hook(layer, input, output):
            if io == 'o':
                if idx == "None":
                    outs[name] = output
                else:
                    outs[name] = output[idx]
            else:
                if idx == "None":
                    outs[name] = input
                else:
                    outs[name] = input[idx]

        return get_output_hook

    ### TODO: support DP model
    for i, (n, m) in enumerate(model.named_sublayers()):
        if n == layers_name:
            m.register_forward_post_hook(
                _get_activation(outs, hook_layers_name, io, idx))


class Distill(nn.Layer):
    """
        Distill API.
        distill_configs(list(dict) | path): the list of distill config.
        student_models(list(nn.Layer)): the list of student model, the state of student model must be training mode.
        teacher_models(list(nn.Layer)): the list of teacher model.
        return_model_outputs(bool): whether to return model output. Default: True.
    """

    def __init__(self,
                 distill_configs,
                 student_models,
                 teacher_models,
                 return_model_outputs=True):
        super(Distill, self).__init__()
        if isinstance(student_models, nn.Layer):
            student_models = [student_models]
        if isinstance(teacher_models, nn.Layer):
            teacher_models = [teacher_models]

        if isinstance(distill_configs, list):
            self._distill_configs = distill_configs
        elif os.path.exists(distill_configs):
            if distill_configs.endswith(".yaml"):
                self._distill_configs = yaml2config(distill_configs)
            else:
                raise NotImplementedError("distill config file type error!")
        else:
            raise NotImplementedError("distill config error!")
        self._student_models = nn.LayerList(student_models)
        self._teacher_models = nn.LayerList(teacher_models)
        self._return_model_outputs = return_model_outputs

        self._loss_config_list = []
        for c in self._distill_configs:
            self._transpose_config(c)

        self._hook_layers = self._extract_hook_position()

        # use self._loss_config_list to create all loss object
        self.distill_loss = losses.CombinedLoss(self._loss_config_list)

        self._output_tensor_dict = self._prepare_outputs()
        self._check_hook_output = False

    def parameters(self):
        return self._student_models.parameters() + self.distill_loss.parameters(
        )

    def _extract_hook_position(self):
        """ extrat hook position according to config"""
        model_hook_layers = {}
        for config in self._loss_config_list:
            model_name_pairs = config['model_name_pairs']
            layers_name = config['layers_name']
            for model_name_pair in model_name_pairs:
                for idx, model_name in enumerate(model_name_pair):
                    if model_name not in model_hook_layers:
                        model_hook_layers[model_name] = [layers_name[idx]]
                    else:
                        model_hook_layers[model_name].append(layers_name[idx])
        for model_name, hook_layers in model_hook_layers.items():
            model_hook_layers[model_name] = list(set(hook_layers))
        return model_hook_layers

    def _transpose_config(self, config):
        """ Transpose config to loss needed """
        global_config = {}
        if 'model_name_pairs' not in config:
            global_config['model_name_pairs'] = [['student_0', 'teacher_0']]
        else:
            if isinstance(config['model_name_pairs'][0], str):
                config['model_name_pairs'] = [config['model_name_pairs']]
            global_config['model_name_pairs'] = config['model_name_pairs']
            config.pop('model_name_pairs')

        for key in config.keys():
            if key != 'layers':
                global_config[key] = config[key]

        for per_layer_config in config['layers']:
            per_layer_config.update(copy.deepcopy(global_config))
            layer_config = LayerConfig(**per_layer_config).__dict__
            for idx in range(len(layer_config['layers_name'])):
                ### slice 0 from string "input" or "output", results is "i" or "o".
                postfix = '#' + layer_config['io'][idx][0] + '#' + str(
                    layer_config['idx'][idx])
                layer_config['layers_name'][idx] += postfix
            ### io and idx only use to extract tensor from hook, so pop it here.
            layer_config.pop('io')
            layer_config.pop('idx')
            self._loss_config_list.append(layer_config)

    def _prepare_outputs(self):
        """
        Add hook to get the output tensor of target layer.
        """
        outputs_tensor = {}
        for idx, m in enumerate(self._student_models):
            hook_layers = self._hook_layers['student_{}'.format(idx)]
            stu_outs = collections.OrderedDict()
            outputs_tensor['student_{}'.format(idx)] = self._prepare_hook(
                m, hook_layers, stu_outs)
        for idx, m in enumerate(self._teacher_models):
            hook_layers = self._hook_layers['teacher_{}'.format(idx)]
            tea_outs = collections.OrderedDict()
            outputs_tensor['teacher_{}'.format(idx)] = self._prepare_hook(
                m, hook_layers, tea_outs)
        return outputs_tensor

    def _prepare_hook(self, model, hook_layers, outs_dict):
        """
        Add hook.
        """
        for layer in hook_layers:
            tmp = layer.strip().split('#')
            layer_name, io, idx = tmp[0], tmp[1], tmp[2]
            if idx != "None":
                idx = int(idx)
            _add_hooks(model, outs_dict, layer_name, layer, io, idx)
        return outs_dict

    def forward(self, *inputs, **kwargs):
        students_batch_outs = []
        teachers_batch_outs = []
        for idx, student_model in enumerate(self._student_models):
            stu_batch_outs = student_model.forward(*inputs, **kwargs)
            students_batch_outs.append(stu_batch_outs)
        for idx, teacher_model in enumerate(self._teacher_models):
            tea_batch_outs = teacher_model.forward(*inputs, **kwargs)
            if not teacher_model.training:
                tea_batch_outs = [i.detach() for i in tea_batch_outs]
            teachers_batch_outs.extend(tea_batch_outs)

        if len(self._student_models) == 1:
            students_batch_outs = students_batch_outs[0]
        if len(self._teacher_models) == 1:
            teachers_batch_outs = teachers_batch_outs[0]

        if self._check_hook_output is False:
            self._check_hook_output = True
            for mo, hook_out in self._output_tensor_dict.items():
                for hook_name, hook_value in hook_out.items():
                    hook_name = hook_name.strip().split('#')[0]
                    assert type(hook_value) is paddle.Tensor or len(\
                        hook_value) == 1, \
                        "model: {} layer: {} has more than one output/input" \
                        ", please specific the idx of output/input.".format(mo, hook_name)
        ### batch is None just for now
        distill_outputs = self.distill_loss(self._output_tensor_dict, None)
        distill_loss = distill_outputs['loss']

        if self._return_model_outputs:
            return distill_loss, students_batch_outs, teachers_batch_outs
        else:
            return distill_loss
