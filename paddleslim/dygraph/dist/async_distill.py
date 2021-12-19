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
import time
import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.fluid.dygraph.parallel import sync_params_buffers
from paddle.distributed.fleet.utils.hybrid_parallel_util import _apply_collective_grads as apply_collective_grads

from .distill import Distill, remove_hooks
from ...common.wrapper_function import init_index

__all__ = ['AsyncDistill']


class OptimizerWrapper(object):
    def __init__(self, opt, parameters, group):
        self.opt = opt
        self.parameters = parameters
        self.group = group

    def step(self):
        print("this is wrapper step")
        with paddle.no_grad():
            apply_collective_grads(self.parameters, self.group)
        self.opt.step()


class AsyncDistill(Distill):
    def __init__(self,
                 configs,
                 students,
                 teachers,
                 student_dev_id,
                 teacher_dev_id,
                 optimizer,
                 convert_fn=True):
        self._student_dev_id = student_dev_id
        self._teacher_dev_id = teacher_dev_id
        self._all_dev_id = self._student_dev_id + self._teacher_dev_id
        self._device_id = dist.get_rank()
        self._student_gp = dist.new_group(self._student_dev_id)
        self._bd_dp = dist.new_group(self._all_dev_id)

        super(AsyncDistill, self).__init__(configs, students, teachers)
        self.optimizer = OptimizerWrapper(optimizer,
                                          self.parameters(), self._student_gp)

        ### TODO: support more than one student
        assert len(self._student_models) == 1, "only support one student now"
        ### NOTE: only support one teacher in one gpu.
        assert len(self._teacher_dev_id) == len(
            self._teacher_models
        ), "the number of teacher must equal to the number of teacher_dev_id"

        ### Synchronize the parameter of student
        sync_params_buffers(
            self._student_models[0], comm_group=self._student_gp)

    def _initialize_hook(self):
        if self._device_id in self._teacher_dev_id:
            self._teacher_output_tensor_dict = self._prepare_outputs(self._hook_layers, \
                  self._teacher_models, False, self._teacher_output_tensor_dict)
        else:
            self._student_output_tensor_dict = self._prepare_outputs(self._hook_layers, \
                  self._student_models, True, self._student_output_tensor_dict)

    def _inference_teacher_model(self, *inputs, **kwargs):
        ### function convert to class in model after useless forward, so hook 
        ### in function only can be add here and ```in_forward``` set to True.
        update_teacher_output_tensor_dict = {}
        update_teacher_output_tensor_dict = self._prepare_outputs(
            self._hook_layers,
            self._teacher_models,
            False,
            update_teacher_output_tensor_dict,
            in_forward=True)

        teachers_batch_outs = []
        for idx, teacher_model in enumerate(self._teacher_models):
            ### initialize global index before each forward
            init_index()
            tea_batch_outs = teacher_model.forward(*inputs, **kwargs)

            if not teacher_model.training:
                tea_batch_outs = [i.detach() for i in tea_batch_outs]
            teachers_batch_outs.extend(tea_batch_outs)
        if len(self._teacher_models) == 1:
            teachers_batch_outs = teachers_batch_outs[0]

        print("self._teacher_output_tensor_dict: ",
              self._teacher_output_tensor_dict)
        #print("self._teacher_output_tensor_dict: ", self._student_output_tensor_dict)

        for model, _ in self._teacher_output_tensor_dict.items():
            self._teacher_output_tensor_dict[model].update(
                update_teacher_output_tensor_dict[model])

        ### scatter inputs to student 
        ### TODO: support multi-teacher need to change here.
        for inp in inputs:
            tmp_inp = paddle.zeros(inp.shape)
            inp_list = paddle.split(
                inp, num_or_sections=len(self._student_dev_id), axis=0)
            inp_list.insert(self._device_id, inp_list[0])
            dist.scatter(
                tmp_inp,
                tensor_list=inp_list,
                src=self._device_id,
                group=self._bd_dp)

        ### scatter kwargs to student, scatter only support paddle.Tensor.
        for _, value in kwargs.items():
            if isinstance(value, paddle.Tensor):
                tmp_value = paddle.zeros(value.shape)
                value_list = paddle.split(
                    value, num_or_sections=len(self._student_dev_id), axis=0)
                value_list.insert(self._device_id, value_list[0])
                dist.scatter(
                    tmp_value,
                    tensor_list=value_list,
                    src=self._device_id,
                    group=self._bd_dp)

        for inter_tensor_name, inter_tensor in self._teacher_output_tensor_dict[\
               'teacher_{}'.format(self._teacher_dev_id.index(self._device_id))].items():
            tmp_tensor = paddle.zeros(inter_tensor.shape)
            tensor_list = paddle.split(
                inter_tensor, num_or_sections=len(self._student_dev_id), axis=0)
            tensor_list.insert(self._device_id, tensor_list[0])
            print(tensor_list)
            dist.scatter(
                tmp_tensor,
                tensor_list=tensor_list,
                src=self._device_id,
                group=self._bd_dp)

        return teachers_batch_outs

    def _train_student_model(self):
        inputs = []
        kwargs = dict()
        for tea_id in self._teacher_dev_id:
            for inp_shape, inp_dtype in zip(self.input_tensor_shape,
                                            self.input_tensor_dtype):
                tmp_inputs = paddle.zeros([1, 3, 32, 32], dtype=inp_dtype)
                dist.scatter(tmp_inputs, src=tea_id, group=self._bd_dp)
                inputs.append(tmp_inputs)
                del tmp_inputs

            for kw_name, kw_shape, kw_dtype in zip(
                    self.kwargs_name, self.kwargs_shape, self.kwargs_dtype):
                tmp_inputs = paddle.zeros(kw_shape, dtype=kw_dtype)
                dist.scatter(tmp_inputs, src=tea_id, group=self._bd_dp)
                kwargs[kw_name] = tmp_inputs
                del tmp_inputs

        kwargs.update(self.other_kwargs)

        ### function convert to class in model after useless forward, so hook 
        ### in function only can be add here and ```in_forward``` set to True.
        update_student_output_tensor_dict = {}
        update_student_output_tensor_dict = self._prepare_outputs(
            self._hook_layers,
            self._student_models,
            True,
            update_student_output_tensor_dict,
            in_forward=True)

        students_batch_outs = []
        for idx, student_model in enumerate(self._student_models):
            ### initialize global index before each forward
            init_index()
            stu_batch_outs = student_model.forward(*inputs, **kwargs)
            students_batch_outs.append(stu_batch_outs)

        for model, _ in self._student_output_tensor_dict.items():
            self._student_output_tensor_dict[model].update(
                update_student_output_tensor_dict[model])

        for tea_id in self._teacher_dev_id:
            tmp_inputs = paddle.zeros([1, 3, 32, 32])
            dist.scatter(tmp_inputs, src=tea_id, group=self._bd_dp)
            print(tmp_inputs)
            inputs.append(tmp_inputs)
            del tmp_inputs

        #print("self._student_output_tensor_dict: ", self._student_output_tensor_dict)
        #print("self._student_output_tensor_dict: ", self._teacher_output_tensor_dict)
        #distill_outputs = self.distill_loss(self._output_tensor_dict, None)
        #distill_loss = distill_outputs['loss']
        distill_loss = paddle.mean(students_batch_outs[0])

        remove_hooks(self.forward_hooks)

        return distill_loss, students_batch_outs

    def forward(self, *inputs, **kwargs):
        self.input_tensor_shape = []
        self.input_tensor_dtype = []
        for inp in inputs:
            self.input_tensor_shape.append(inp.shape)
            self.input_tensor_dtype.append(inp.dtype)

        self.kwargs_name = []
        self.kwargs_shape = []
        self.kwargs_dtype = []
        self.other_kwargs = dict()
        for name, value in kwargs.items():
            if isinstance(value, paddle.Tensor):
                self.kwargs_name.append(name)
                self.kwargs_shape.append(name)
                self.kwargs_dtype.append(value.dtype)
            else:
                self.other_kwargs[name] = value

        if self._device_id in self._teacher_dev_id:
            print("this is teacher id: ", self._device_id)
            if self._check_teacher_hook_output is False:
                ### the first useless forward is to convert function to class. 
                self._useless_teacher_forward(*inputs, **kwargs)

            self._inference_teacher_model(*inputs, **kwargs)

            if self._check_teacher_hook_output is False:
                self._check_teacher_hook_output = True
                self._check_output_dict(self._teacher_output_tensor_dict)
            #return None
        else:
            print("this is student id: ", self._device_id)
            if self._check_student_hook_output is False:
                ### the first useless forward is to convert function to class. 
                self._useless_student_forward(*inputs, **kwargs)

            distill_loss, _ = self._train_student_model()

            if self._check_student_hook_output is False:
                self._check_student_hook_output = True
                self._check_output_dict(self._student_output_tensor_dict)

            return distill_loss
