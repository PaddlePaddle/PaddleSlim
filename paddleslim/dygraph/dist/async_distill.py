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


def str_to_tensor(string):
    return paddle.to_tensor([ord(c) for c in string])


def tensor_to_str(tensor):
    return ''.join(chr(i) for i in tensor.numpy())


class OptimizerWrapper(object):
    def __init__(self, opt, parameters, group):
        self.opt = opt
        self.parameters = parameters
        self.group = group

    def step(self):
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
        self._device_id = dist.get_rank()
        self._initialize_groups()

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

    def _initialize_groups(self):
        self._student_gp = dist.new_group(self._student_dev_id)
        for idx, _ in enumerate(self._teacher_dev_id):
            setattr(self, '_st{}_gp'.format(idx),
                    dist.new_group(self._student_dev_id +
                                   [self._teacher_dev_id[idx]]))

        if len(self._teacher_dev_id) == 1:
            ### new groups between teacher and student
            self._teacher_gp = None
        else:
            ### new groups between teachers, new groups between each teacher and student
            self._teacher_gp = dist.new_group(self._teacher_dev_id)

    def _initialize_hook(self):
        if self._device_id in self._teacher_dev_id:
            idx = self._teacher_dev_id.index(self._device_id)
            self._teacher_output_tensor_dict = self._prepare_outputs(self._hook_layers, \
              [self._teacher_models[idx]], False, self._teacher_output_tensor_dict, index=idx)
        else:
            self._student_output_tensor_dict = self._prepare_outputs(self._hook_layers, \
                  self._student_models, True, self._student_output_tensor_dict)

    def _inference_teacher_model(self, *inputs, **kwargs):
        teacher_id = self._teacher_dev_id.index(self._device_id)
        teacher_model = self._teacher_models[teacher_id]
        st_gp = getattr(self, '_st{}_gp'.format(teacher_id))
        ### function convert to class in model after useless forward, so hook 
        ### in function only can be add here and ```in_forward``` set to True.
        update_teacher_output_tensor_dict = {}
        update_teacher_output_tensor_dict = self._prepare_outputs(
            self._hook_layers, [teacher_model],
            False,
            update_teacher_output_tensor_dict,
            index=teacher_id,
            in_forward=True)

        ### all_gather all inputs from multi-teacher
        tmp_inputs = []
        tmp_kwargs = dict()
        tmp_inp = []
        tmp_value = []
        if self._teacher_gp != None:
            for inp in inputs:
                dist.all_gather(tmp_inp, inp, group=self._teacher_gp)
                tmp_inputs.append(paddle.concat(tmp_inp, axis=0))
                tmp_inp = []

            for name, value in kwargs.items():
                if isinstance(value, paddle.Tensor):
                    dist.all_gather(tmp_value, value, group=self._teacher_gp)
                    tmp_kwargs[name] = paddle.concat(tmp_value, axis=0)
                    tmp_value = []
        else:
            tmp_inputs = inputs
            tmp_kwargs = kwargs

        ### initialize global index before each forward
        init_index()
        tea_batch_outs = teacher_model.forward(*tmp_inputs, **tmp_kwargs)
        if not teacher_model.training:
            tea_batch_outs = [i.detach() for i in tea_batch_outs]

        for model, _ in self._teacher_output_tensor_dict.items():
            self._teacher_output_tensor_dict[model].update(
                update_teacher_output_tensor_dict[model])

        ### send the number of inputs to student from teacher 0
        ### after all gather, all teacher has whole inputs
        if teacher_id == 0:
            dist.broadcast(
                paddle.to_tensor(len(tmp_inputs)),
                src=self._device_id,
                group=st_gp)
            for inp in tmp_inputs:
                shapes = []
                shapes.append(len(inp.shape))
                shapes.append(len(str(inp.dtype).split('.')[-1]))
                ### send the length and dtype of inputs to student
                dist.broadcast(
                    paddle.to_tensor(np.array(shapes)),
                    src=self._device_id,
                    group=st_gp)

                inp_list = paddle.split(
                    inp, num_or_sections=len(self._student_dev_id), axis=0)

                ### send the shape of inputs to student
                dist.broadcast(
                    paddle.shape(inp_list[0]), src=self._device_id, group=st_gp)
                ### send the dtype of inputs to student
                dist.broadcast(
                    str_to_tensor(str(inp.dtype).split('.')[-1]),
                    src=self._device_id,
                    group=st_gp)

                tmp_inp = paddle.zeros(inp.shape)
                inp_list.insert(self._device_id, inp_list[0])
                ### scatter inputs to student 
                dist.scatter(
                    tmp_inp,
                    tensor_list=inp_list,
                    src=self._device_id,
                    group=st_gp)

            kwargs_tensor = 0
            for name, value in kwargs.items():
                if isinstance(value, paddle.Tensor):
                    kwargs_tensor += 1
            ### send the number of kwargs to student
            dist.broadcast(
                paddle.to_tensor(kwargs_tensor),
                src=self._device_id,
                group=st_gp)
            for name, value in kwargs.items():
                if isinstance(value, paddle.Tensor):
                    shapes = []
                    shapes.append(len(name))
                    shapes.append(len(value.shape))
                    shapes.append(len(str(value.dtype).split('.')[-1]))
                    ### send length of name and length of shape and length of dtype to student
                    dist.broadcast(
                        paddle.to_tensor(np.array(shapes)),
                        src=self._device_id,
                        group=st_gp)

                    value_list = paddle.split(
                        value,
                        num_or_sections=len(self._student_dev_id),
                        axis=0)

                    ### send name and shape and dtype to student
                    dist.broadcast(
                        str_to_tensor(name), src=self._device_id, group=st_gp)
                    dist.broadcast(
                        paddle.shape(value_list[0]),
                        src=self._device_id,
                        group=st_gp)
                    dist.broadcast(
                        str_to_tensor(str(value.dtype).split('.')[-1]),
                        src=self._device_id,
                        group=st_gp)

                    tmp_value = paddle.zeros(value.shape)
                    value_list.insert(self._device_id, value_list[0])
                    ### scatter kwargs to student, scatter only support paddle.Tensor.
                    dist.scatter(
                        tmp_value,
                        tensor_list=value_list,
                        src=self._device_id,
                        group=st_gp)

        ### send length of intermediate to student
        ### send intermediate tensor from all teacher
        if len(self._teacher_output_tensor_dict) == 0:
            length_output_dict = 0
        else:
            length_output_dict = len(self._teacher_output_tensor_dict[
                'teacher_{}'.format(teacher_id)])
        dist.broadcast(
            paddle.to_tensor(length_output_dict),
            src=self._device_id,
            group=st_gp)
        if length_output_dict > 0:
            for inter_tensor_name, inter_tensor in self._teacher_output_tensor_dict[\
                   'teacher_{}'.format(teacher_id)].items():
                shapes = []
                shapes.append(len(inter_tensor_name))
                shapes.append(len(inter_tensor.shape))
                ### send length of name and length of shape to student
                dist.broadcast(
                    paddle.to_tensor(np.array(shapes)),
                    src=self._device_id,
                    group=st_gp)

                tensor_list = paddle.split(
                    inter_tensor,
                    num_or_sections=len(self._student_dev_id),
                    axis=0)
                ### send name and shape to student
                dist.broadcast(
                    str_to_tensor(inter_tensor_name),
                    src=self._device_id,
                    group=st_gp)
                dist.broadcast(
                    paddle.shape(tensor_list[0]),
                    src=self._device_id,
                    group=st_gp)

                tmp_tensor = paddle.zeros(inter_tensor.shape)
                ### scatter tensor to student
                tensor_list.insert(self._device_id, tensor_list[0])
                dist.scatter(
                    tmp_tensor,
                    tensor_list=tensor_list,
                    src=self._device_id,
                    group=st_gp)

        #return teachers_batch_outs

    def _train_student_model(self, **kwargs):
        inputs = []
        tmp_kwargs = dict()
        input_number = paddle.zeros([1], dtype='int64')
        ### recv the number of inputs
        ### only recv inputs from teacher 0
        dist.broadcast(
            input_number, src=self._teacher_dev_id[0], group=self._st0_gp)
        input_num = input_number.numpy()
        for _ in range(input_num[0]):
            ### recv the length and dtype of inputs
            tmp_inputs_shapes = paddle.zeros([2], dtype='int64')
            dist.broadcast(
                tmp_inputs_shapes,
                src=self._teacher_dev_id[0],
                group=self._st0_gp)

            ### recv the shape of inputs
            tmp_shape = paddle.zeros(
                [tmp_inputs_shapes.numpy()[0]], dtype='int32')
            dist.broadcast(
                tmp_shape, src=self._teacher_dev_id[0], group=self._st0_gp)

            ### recv the dtype of inputs
            tmp_dtype = paddle.zeros(
                [tmp_inputs_shapes.numpy()[1]], dtype='int64')
            dist.broadcast(
                tmp_dtype, src=self._teacher_dev_id[0], group=self._st0_gp)

            ### recv the inputs
            tmp_inputs = paddle.zeros(tmp_shape, dtype=tensor_to_str(tmp_dtype))
            dist.scatter(
                tmp_inputs, src=self._teacher_dev_id[0], group=self._st0_gp)
            inputs.append(tmp_inputs)
            del tmp_inputs

        ### recv the number of kwargs
        dist.broadcast(
            input_number, src=self._teacher_dev_id[0], group=self._st0_gp)
        input_num = input_number.numpy()
        for _ in range(input_num[0]):
            ### recv length of name and length of shape and length of dtype
            tmp_inputs_shapes = paddle.zeros([3], dtype='int64')
            dist.broadcast(
                tmp_inputs_shapes,
                src=self._teacher_dev_id[0],
                group=self._st0_gp)

            ### recv the name of kwargs
            tmp_name = paddle.zeros(
                [tmp_inputs_shapes.numpy()[0]], dtype='int64')
            dist.broadcast(
                tmp_name, src=self._teacher_dev_id[0], group=self._st0_gp)
            ### recv the shape of kwargs 
            tmp_shape = paddle.zeros(
                [tmp_inputs_shapes.numpy()[1]], dtype='int32')
            dist.broadcast(
                tmp_shape, src=self._teacher_dev_id[0], group=self._st0_gp)
            ### recv the dtype of kwargs 
            tmp_dtype = paddle.zeros(
                [tmp_inputs_shapes.numpy()[2]], dtype='int64')
            dist.broadcast(
                tmp_dtype, src=self._teacher_dev_id[0], group=self._st0_gp)

            ### recv the kwargs tensor
            tmp_inputs = paddle.zeros(tmp_shape, dtype=tensor_to_str(tmp_dtype))
            dist.scatter(
                tmp_inputs, src=self._teacher_dev_id[0], group=self._st0_gp)
            tmp_kwargs[tensor_to_str(tmp_name)] = tmp_inputs
            del tmp_inputs

        ### update kwargs from student model
        kwargs.update(tmp_kwargs)

        if self._check_student_hook_output is False:
            ### the first useless forward is to convert function to class. 
            self._useless_student_forward(*inputs, **kwargs)

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

        teacher_output_tensor_dict = dict()
        for idx, tea_id in enumerate(self._teacher_dev_id):
            st_gp = getattr(self, '_st{}_gp'.format(idx))

            ### recv length of intermediate 
            dist.broadcast(input_number, src=tea_id, group=st_gp)
            input_num = input_number.numpy()
            for _ in range(input_num[0]):
                ### recv length of name and length of shape
                tmp_inputs_shapes = paddle.zeros([2], dtype='int64')
                dist.broadcast(tmp_inputs_shapes, src=tea_id, group=st_gp)

                ### recv the name of intermediate tensor
                tmp_name = paddle.zeros(
                    [tmp_inputs_shapes.numpy()[0]], dtype='int64')
                dist.broadcast(tmp_name, src=tea_id, group=st_gp)
                ### recv the shape of intermediate tensor
                tmp_shape = paddle.zeros(
                    [tmp_inputs_shapes.numpy()[1]], dtype='int32')
                dist.broadcast(tmp_shape, src=tea_id, group=st_gp)

                ### recv the intermediate tensor
                tmp_inter = paddle.zeros(tmp_shape)
                dist.scatter(tmp_inter, src=tea_id, group=st_gp)
                teacher_name = 'teacher_{}'.format(
                    self._teacher_dev_id.index(tea_id))
                if teacher_name not in teacher_output_tensor_dict:
                    teacher_output_tensor_dict[
                        teacher_name] = collections.OrderedDict()
                teacher_output_tensor_dict[teacher_name][tensor_to_str(
                    tmp_name)] = tmp_inter
                del tmp_inter

        if self._check_student_hook_output is False:
            self._check_student_hook_output = True
            self._check_output_dict(self._student_output_tensor_dict)

        output_tensor_dict = dict(self._student_output_tensor_dict,
                                  **teacher_output_tensor_dict)
        distill_outputs = self.distill_loss(output_tensor_dict, None)
        distill_loss = distill_outputs['loss']

        remove_hooks(self.forward_hooks)

        return distill_loss, students_batch_outs

    def forward(self, *inputs, **kwargs):
        if self._device_id in self._teacher_dev_id:
            print("this is teacher id: ", self._device_id,
                  self._check_teacher_hook_output)
            if self._check_teacher_hook_output is False:
                ### the first useless forward is to convert function to class. 
                self._useless_teacher_forward(
                    *inputs,
                    **kwargs,
                    teacher_idx=self._teacher_dev_id.index(self._device_id))

            self._inference_teacher_model(*inputs, **kwargs)

            if self._check_teacher_hook_output is False:
                self._check_teacher_hook_output = True
                self._check_output_dict(self._teacher_output_tensor_dict)

        else:
            distill_loss, _ = self._train_student_model(**kwargs)

            return distill_loss
