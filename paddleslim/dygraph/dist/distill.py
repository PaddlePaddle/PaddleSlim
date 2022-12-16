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
from ...common.wrapper_function import init_index, functional2layer
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
            hooks = m.register_forward_post_hook(
                _get_activation(outs, hook_layers_name, io, idx))
    return hooks


def _remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


class Distill(paddle.nn.Layer):
    """
        Distill API.
        configs(list(dict) | string): the list of distill config or the path of yaml file which contain the distill config.
        students(list(nn.Layer)): the list of student model, the state of student model must be training mode.
        teachers(list(nn.Layer)): the list of teacher model.
        convert_fn(bool): convert the functional in paddlepaddle to nn.Layer. The detail of this convert operation please 
                          reference to ```paddleslim.common.functional2layer```. Default: True.
        return_model_outputs(bool): whether to return the origin outputs of the model. If set to True, will return distill loss, the output of students and the output of teachers, the output of each part will be returned as a list. Default: True.
    """

    def __init__(self,
                 configs,
                 students,
                 teachers,
                 convert_fn=True,
                 return_model_outputs=True):
        super(Distill, self).__init__()
        if convert_fn:
            functional2layer()
        if isinstance(students, paddle.nn.Layer):
            students = [students]
        if isinstance(teachers, paddle.nn.Layer):
            teachers = [teachers]

        if isinstance(configs, list):
            self._configs = configs
        elif os.path.exists(configs):
            if configs.endswith(".yaml"):
                self._configs = yaml2config(configs)
            else:
                raise NotImplementedError("distill config file type error!")
        else:
            raise NotImplementedError("distill config error!")
        self._student_models = paddle.nn.LayerList(students)
        self._teacher_models = paddle.nn.LayerList(teachers)
        self._return_model_outputs = return_model_outputs

        self._loss_config_list = []
        for c in self._configs:
            unfold_layer_config = self._transpose_config(c)
            self._loss_config_list.extend(unfold_layer_config)

        hook_layers = self._extract_hook_position()
        self._hook_layers = hook_layers

        # use self._loss_config_list to create all loss object
        self.distill_loss = losses.CombinedLoss(self._loss_config_list)

        self._output_tensor_dict = self._prepare_outputs(hook_layers)
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
        unfold_config = []
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
            unfold_config.append(layer_config)
        return unfold_config

    def _prepare_outputs(self, hook_layers, in_forward=False):
        """
        Add hook to get the output tensor of target layer.
        """
        outputs_tensor = {}
        for idx, m in enumerate(self._student_models):
            tmp_hook_layers = hook_layers['student_{}'.format(idx)]
            stu_outs = collections.OrderedDict()
            outputs_tensor['student_{}'.format(idx)] = self._prepare_hook(
                m, tmp_hook_layers, stu_outs, in_forward=in_forward)
        for idx, m in enumerate(self._teacher_models):
            tmp_hook_layers = hook_layers['teacher_{}'.format(idx)]
            tea_outs = collections.OrderedDict()
            outputs_tensor['teacher_{}'.format(idx)] = self._prepare_hook(
                m, tmp_hook_layers, tea_outs, in_forward=in_forward)
        return outputs_tensor

    def _prepare_hook(self, model, hook_layers, outs_dict, in_forward):
        """
        Add hook.
        """
        self.forward_hooks = []
        for layer in hook_layers:
            tmp = layer.strip().split('#')
            layer_name, io, idx = tmp[0], tmp[1], tmp[2]
            if idx != "None":
                idx = int(idx)
            if in_forward:
                if 'wrap_fn_' in layer_name:
                    hooks = _add_hooks(model, outs_dict, layer_name, layer, io,
                                       idx)
                    self.forward_hooks.append(hooks)
            else:
                if 'wrap_fn_' not in layer_name:
                    _add_hooks(model, outs_dict, layer_name, layer, io, idx)
        return outs_dict

    def _useless_forward(self, *inputs, **kwargs):
        for idx, student_model in enumerate(self._student_models):
            ### initialize global index before each forward
            init_index()
            student_model.forward(*inputs, **kwargs)
        for idx, teacher_model in enumerate(self._teacher_models):
            ### initialize global index before each forward
            init_index()
            teacher_model.forward(*inputs, **kwargs)

    def forward(self, *inputs, **kwargs):
        if self._check_hook_output is False:
            ### the first useless forward is to convert function to class. 
            self._useless_forward(*inputs, **kwargs)

        update_output_tensor_dict = self._prepare_outputs(
            self._hook_layers, in_forward=True)

        students_batch_outs = []
        teachers_batch_outs = []
        for idx, student_model in enumerate(self._student_models):
            ### initialize global index before each forward
            init_index()
            stu_batch_outs = student_model.forward(*inputs, **kwargs)
            students_batch_outs.append(stu_batch_outs)
        for idx, teacher_model in enumerate(self._teacher_models):
            ### initialize global index before each forward
            init_index()
            tea_batch_outs = teacher_model.forward(*inputs, **kwargs)
            if not teacher_model.training:
                tea_batch_outs = [i.detach() for i in tea_batch_outs]
            teachers_batch_outs.extend(tea_batch_outs)

        ### update hook information.
        for model, _ in self._output_tensor_dict.items():
            self._output_tensor_dict[model].update(update_output_tensor_dict[
                model])

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

        _remove_hooks(self.forward_hooks)

        if self._return_model_outputs:
            return distill_loss, students_batch_outs, teachers_batch_outs
        else:
            return distill_loss
