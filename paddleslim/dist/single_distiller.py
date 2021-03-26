# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle


def merge(teacher_program,
          student_program,
          data_name_map,
          place,
          scope=None,
          name_prefix='teacher_'):
    """Merge teacher program into student program and add a uniform prefix to the
    names of all vars in teacher program

    Args:
        teacher_program(Program): The input teacher model paddle program 
        student_program(Program): The input student model paddle program
        data_map_map(dict): Mapping of teacher input interface name and student
                            input interface name, where key of dict is the
                            input name of teacher_program, and value is the
                            input name of student_program.
        place(CPUPlace()|CUDAPlace(N)): This parameter represents
                                                    paddle run on which device.
        scope(Scope): This parameter indicates the variable scope used by
                      the program. If not specified, the default global scope
                      will be used. Default: None
        name_prefix(str): Name prefix added for all vars of the teacher program.
                          Default: 'teacher_'

    Returns:
        None
    """
    if scope == None:
        scope = paddle.static.global_scope()
    teacher_program = teacher_program.clone(for_test=True)
    for teacher_var in teacher_program.list_vars():
        skip_rename = False
        if teacher_var.name != 'fetch' and teacher_var.name != 'feed':
            if teacher_var.name in data_name_map.keys():
                new_name = data_name_map[teacher_var.name]
                if new_name == teacher_var.name:
                    skip_rename = True
            else:
                new_name = name_prefix + teacher_var.name
            if not skip_rename:
                # scope var rename
                old_var = scope.var(teacher_var.name).get_tensor()
                renamed_var = scope.var(new_name).get_tensor()
                renamed_var.set(np.array(old_var), place)

                # program var rename
                renamed_var = teacher_program.global_block()._rename_var(
                    teacher_var.name, new_name)

    for teacher_var in teacher_program.list_vars():
        if teacher_var.name != 'fetch' and teacher_var.name != 'feed':
            # student program add var
            new_var = student_program.global_block()._clone_variable(
                teacher_var, force_persistable=False)
            new_var.stop_gradient = True

    for block in teacher_program.blocks:
        for op in block.ops:
            if op.type != 'feed' and op.type != 'fetch':
                inputs = {}
                outputs = {}
                attrs = {}
                for input_name in op.input_names:
                    inputs[input_name] = [
                        block.var(in_var_name)
                        for in_var_name in op.input(input_name)
                    ]

                for output_name in op.output_names:
                    outputs[output_name] = [
                        block.var(out_var_name)
                        for out_var_name in op.output(output_name)
                    ]
                for attr_name in op.attr_names:
                    attrs[attr_name] = op.attr(attr_name)
                student_program.global_block().append_op(
                    type=op.type, inputs=inputs, outputs=outputs, attrs=attrs)


def fsp_loss(teacher_var1_name,
             teacher_var2_name,
             student_var1_name,
             student_var2_name,
             program=None):
    """Combine variables from student model and teacher model by fsp-loss.

    Args:
        teacher_var1_name(str): The name of teacher_var1.
        teacher_var2_name(str): The name of teacher_var2. Except for the
            second dimension, all other dimensions should
            be consistent with teacher_var1.
        student_var1_name(str): The name of student_var1.
        student_var2_name(str): The name of student_var2. Except for the
            second dimension, all other dimensions should
            be consistent with student_var1.
        program(Program): The input distiller program. If not specified,
                          the default program will be used. Default: None

    Returns:
        Variable: fsp distiller loss.
    """
    if program == None:
        program = paddle.static.default_main_program()
    teacher_var1 = program.global_block().var(teacher_var1_name)
    teacher_var2 = program.global_block().var(teacher_var2_name)
    student_var1 = program.global_block().var(student_var1_name)
    student_var2 = program.global_block().var(student_var2_name)
    teacher_fsp_matrix = paddle.fluid.layers.fsp_matrix(teacher_var1,
                                                        teacher_var2)
    student_fsp_matrix = paddle.fluid.layers.fsp_matrix(student_var1,
                                                        student_var2)
    fsp_loss = paddle.mean(
        paddle.nn.functional.square_error_cost(student_fsp_matrix,
                                               teacher_fsp_matrix))
    return fsp_loss


def l2_loss(teacher_var_name, student_var_name, program=None):
    """Combine variables from student model and teacher model by l2-loss.

    Args:
        teacher_var_name(str): The name of teacher_var.
        student_var_name(str): The name of student_var.
        program(Program): The input distiller program. If not specified,
                          the default program will be used. Default: None

    Returns: 
        Variable: l2 distiller loss.
    """
    if program == None:
        program = paddle.static.default_main_program()
    student_var = program.global_block().var(student_var_name)
    teacher_var = program.global_block().var(teacher_var_name)
    l2_loss = paddle.mean(
        paddle.nn.functional.square_error_cost(student_var, teacher_var))
    return l2_loss


def soft_label_loss(teacher_var_name,
                    student_var_name,
                    program=None,
                    teacher_temperature=1.,
                    student_temperature=1.):
    """Combine variables from student model and teacher model by soft-label-loss.

    Args:
        teacher_var_name(str): The name of teacher_var.
        student_var_name(str): The name of student_var.
        program(Program): The input distiller program. If not specified,
                          the default program will be used. Default: None
        teacher_temperature(float): Temperature used to divide
            teacher_feature_map before softmax. Default: 1.0
        student_temperature(float): Temperature used to divide 
            student_feature_map before softmax. Default: 1.0

    Returns:
        Variable: l2 distiller loss.
    """
    if program == None:
        program = paddle.static.default_main_program()
    student_var = program.global_block().var(student_var_name)
    teacher_var = program.global_block().var(teacher_var_name)
    teacher_var.stop_gradient = True

    student_var = paddle.nn.functional.softmax(student_var /
                                               student_temperature)
    teacher_var = paddle.nn.functional.softmax(teacher_var /
                                               teacher_temperature)
    soft_label_loss = paddle.mean(
        paddle.fluid.layers.cross_entropy(
            student_var, teacher_var, soft_label=True))
    return soft_label_loss


def loss(loss_func, program=None, **kwargs):
    """Combine variables from student model and teacher model by self defined loss.

    Args:
        program(Program): The input distiller program. If not specified,
                          the default program will be used. Default: None
        loss_func(function): The user self defined loss function. 

    Returns: 
        Variable: self defined distiller loss.
    """
    if program == None:
        program = paddle.static.default_main_program()
    func_parameters = {}
    for item in kwargs.items():
        if isinstance(item[1], str):
            func_parameters.setdefault(item[0],
                                       program.global_block().var(item[1]))
        else:
            func_parameters.setdefault(item[0], item[1])
    loss = loss_func(**func_parameters)
    return loss

def hard_distill_loss(teacher_var_name,
                      student_var_name,
                      program,
                      batch_size=32):
    """ 
     Hard sample distill loss
     Args:
         teacher_var_name(str): name of teacher_var
         student_var_name(str): name of student_var
         Program: The input distiller program. If not specified,
                  the default program will be used. Default: None
         batch_size: batchsize
     Returns:
         Variable: loss of hard sample
    """
    student_var = program.global_block().var(student_var_name)
    teacher_var = program.global_block().var(teacher_var_name)
    teacher_var = fluid.layers.reshape(teacher_var, [batch_size, -1])
    student_var = fluid.layers.reshape(student_var, [batch_size, -1])
    norm_teacher = fluid.layers.sqrt(
            fluid.layers.reduce_sum(
                 fluid.layers.square(teacher_var), dim=1))
    norm_teacher = fluid.layers.elementwise_div(
            teacher_var, norm_teacher, axis=0)

    norm_teacher_T = fluid.layers.transpose(norm_teacher, perm=[1, 0])

    norm_student = fluid.layers.sqrt(
            fluid.layers.reduce_sum(
                      fluid.layers.square(student_var), dim=1))
    norm_student = fluid.layers.elementwise_div(
            student_var, norm_student, axis=0)
    norm_student_T = fluid.layers.transpose(norm_student, perm=[1, 0])

    cos_t = fluid.layers.abs(fluid.layers.mul(norm_teacher, norm_teacher_T))
    cos_s = fluid.layers.abs(fluid.layers.mul(norm_student, norm_student_T))
    diff = fluid.layers.abs(cos_s - cos_t)
    diff = fluid.layers.reshape(diff, [1, -1])

    N = batch_size // 4
    out, idx = fluid.layers.topk(diff, k=N)
    x = idx[0] / batch_size
    y = idx[0] % batch_size
    x = fluid.layers.reshape(x, [N, -1])
    y = fluid.layers.reshape(y, [N, -1])
    index_var = fluid.layers.concat(input=[x, y], axis=1)
    index_var = fluid.layers.reshape(index_var, [-1])
    index_var.stop_gradient = True
    var_t = fluid.layers.gather(teacher_var, index_var)
    var_s = fluid.layers.gather(student_var, index_var)

    loss = fluid.layers.reduce_mean(fluid.layers.square(var_t - var_s))
    return loss

def soft_hard_distill_loss(teacher_var_name,
                           student_var_name,
                           program,
                           batch_size=32):

    """
    Soft hard sample distill loss with tanh function reweight
    Args:
       teacher_var_name(str): The name of teacher_var.
       student_var_name(str): The name of student_var.
       program(Program): The input distiller program. If not specified,
                         the default program will be used. Default: None
       batch_size: batchsize
    Returns:
       variable: loss of soft hard sample
    """

    student_var = program.global_block().var(student_var_name)
    teacher_var = program.global_block().var(teacher_var_name)
    norm_teacher = fluid.layers.sqrt(
                     fluid.layers.reduce_sum(
                           fluid.layers.square(teacher_var), dim=1))
    norm_teacher = fluid.layers.elementwise_div(
                     teacher_var, norm_teacher, axis=0)
    norm_teacher_T = fluid.layers.transpose(norm_teacher, perm=[1, 0])

    norm_student = fluid.layers.sqrt(
                  fluid.layers.reduce_sum(
                         fluid.layers.square(student_var), dim=1))
    norm_student = fluid.layers.elementwise_div(
                    student_var, norm_student, axis=0)
    norm_student_T = fluid.layers.transpose(norm_student, perm=[1, 0])

    cos_t = fluid.layers.abs(fluid.layers.mul(norm_teacher, norm_teacher_T))
    cos_s = fluid.layers.abs(fluid.layers.mul(norm_student, norm_student_T))
    diff = fluid.layers.abs(cos_s - cos_t)
    diff = fluid.layers.reshape(diff, [1, -1])

    N = batch_size // 4 * 3
    out, idx = fluid.layers.argsort(diff, axis=-1)
    x = idx[0] / batch_size
    y = idx[0] % batch_size
    x = fluid.layers.reshape(x, [batch_size, -1])
    y = fluid.layers.reshape(y, [batch_size, -1])

    index_var = fluid.layers.concat(input=[x, y], axis=1)
    index_var = fluid.layers.reshape(index_var, [-1])
    index_var.stop_gradient = True

    var_t = fluid.layers.gather(teacher_var, index_var)
    var_s = fluid.layers.gather(student_var, index_var)

    id_batch = fluid.layers.range(0, batch_size, 1, 'int32')
    weight_batch  = fluid.layers.tanh(id_batch - N) + 1.0

    square_diff = fluid.layers.square(var_t - var_s)
    diff_mean = fluid.layers.reduce_mean(square_diff, axis=-1)
    loss = weight_batch * diff_mean
    return loss

def RK_Angle(teacher_var_name, student_var_name, program, batch_size=32):

    """
    Angle-wise loss in relation knowledge distill loss
    Args:
       teacher_var_name(str): The name of teacher_var.
       student_var_name(str): The name of student_var.
       program(Program): The input distiller program. If not specified,
                         the default program will be used. Default: None
       batch_size: batchsize
    Returns:
       variable: loss of angle-wise loss in relation knowledge distill loss
    """

    student_var = program.global_block().var(student_var_name)
    teacher_var = program.global_block().var(teacher_var_name)

    teacher_a = fluid.layers.unsqueeze(teacher_var, axes=[0])
    teacher_a = fluid.layers.expand(teacher_a, [batch_size, 1, 1])
    teacher_b = fluid.layers.unsqueeze(teacher_var, axes=[1])
    teacher_b = fluid.layers.expand(teacher_b, [1, batch_size, 1])
    teacher_f =  fluid.layers.l2_normalize(teacher_a - teacher_b, axis = 2)
    teacher_f_T = fluid.layers.transpose(teacher_f, perm=[0, 2, 1])
    t_angle = fluid.layers.matmul(teacher_f, teacher_f_T)
    t_angle = fluid.layers.flatten(t_angle, axis=0)

    student_a = fluid.layers.unsqueeze(student_var, axes=[0])
    student_a = fluid.layers.expand(student_a, [batch_size, 1, 1])
    student_b = fluid.layers.unsqueeze(student_var, axes=[1])
    student_b = fluid.layers.expand(student_b, [1, batch_size, 1])
    student_f =  fluid.layers.l2_normalize(student_a - student_b, axis=2)
    student_f_T = fluid.layers.transpose(student_f, perm=[0, 2, 1])
    s_angle = fluid.layers.matmul(student_f, student_f_T)
    s_angle = fluid.layers.flatten(s_angle, axis=0)

    loss = fluid.layers.elementwise_sub(s_angle, t_angle)
    loss = fluid.layers.abs(loss)
    loss = fluid.layers.reduce_mean(loss)
    return loss

def pdist(input, batch_size=32, squared=False, eps=1e-12):
    """
    Function to compute distance
    """
    e_square = fluid.layers.reduce_sum(fluid.layers.square(input), dim=-1)
    e_a = fluid.layers.unsqueeze(e_square, axes=[1])
    e_b = fluid.layers.unsqueeze(e_square, axes=[0])

    input_t = fluid.layers.transpose(input, [1, 0])
    prod = fluid.layers.mul(input, input_t)

    e_a = fluid.layers.expand(e_a, [1, batch_size])
    e_b = fluid.layers.expand(e_b, [batch_size, 1])
    res = e_a + e_b - 2 * prod
    res = fluid.layers.clip(res, min=eps, max=float('inf'))

    if not squared:
           res=fluid.layers.sqrt(res)
    return res

def RK_Distance(teacher_var_name, student_var_name, program, batch_size=32):
    """
    Distance-wise loss in relation knowledge distill loss
    Args:
       teacher_var_name(str): The name of teacher_var.
       student_var_name(str): The name of student_var.
       program(Program): The input distiller program. If not specified,
                         the default program will be used. Default: None
       batch_size: batchsize
    Returns:
       variable: distance-wise loss in relation knowledge distill loss
    """
    teacher_var = program.global_block().var(teacher_var_name)
    student_var = program.global_block().var(student_var_name)

    t_d = pdist(teacher_var, batch_size=batch_size, squared=False)
    mean_td = fluid.layers.reduce_mean(t_d)
    t_d = t_d / (mean_td + 1e-12)
    t_d.stop_gradient = True

    d = pdist(student_var, batch_size = batch_size, squared=False)
    mean_d = fluid.layers.reduce_mean(d)
    d = d / (mean_d + 1e-12)

    loss = fluid.layers.smooth_l1(d, t_d)
    loss = fluid.layers.reduce_mean(loss)
    return loss

