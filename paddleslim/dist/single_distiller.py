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
import paddle.fluid as fluid


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
        place(fluid.CPUPlace()|fluid.CUDAPlace(N)): This parameter represents
                                                    paddle run on which device.
        scope(Scope): This parameter indicates the variable scope used by
                      the program. If not specified, the default global scope
                      will be used. Default: None
        name_prefix(str): Name prefix added for all vars of the teacher program.
                          Default: 'teacher_'

    Returns:
        None
    """
    if scope==None:
        scope = fluid.global_scope()
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
    if program==None:
        program=fluid.default_main_program()
    teacher_var1 = program.global_block().var(teacher_var1_name)
    teacher_var2 = program.global_block().var(teacher_var2_name)
    student_var1 = program.global_block().var(student_var1_name)
    student_var2 = program.global_block().var(student_var2_name)
    teacher_fsp_matrix = fluid.layers.fsp_matrix(teacher_var1, teacher_var2)
    student_fsp_matrix = fluid.layers.fsp_matrix(student_var1, student_var2)
    fsp_loss = fluid.layers.reduce_mean(
        fluid.layers.square(student_fsp_matrix - teacher_fsp_matrix))
    return fsp_loss


def l2_loss(teacher_var_name,
            student_var_name,
            program=None):
    """Combine variables from student model and teacher model by l2-loss.

    Args:
        teacher_var_name(str): The name of teacher_var.
        student_var_name(str): The name of student_var.
        program(Program): The input distiller program. If not specified,
                          the default program will be used. Default: None

    Returns: 
        Variable: l2 distiller loss.
    """
    if program==None:
        program=fluid.default_main_program()
    student_var = program.global_block().var(student_var_name)
    teacher_var = program.global_block().var(teacher_var_name)
    l2_loss = fluid.layers.reduce_mean(
        fluid.layers.square(student_var - teacher_var))
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
    if program==None:
        program=fluid.default_main_program()
    student_var = program.global_block().var(student_var_name)
    teacher_var = program.global_block().var(teacher_var_name)
    student_var = fluid.layers.softmax(student_var / student_temperature)
    teacher_var = fluid.layers.softmax(teacher_var / teacher_temperature)
    teacher_var.stop_gradient = True
    soft_label_loss = fluid.layers.reduce_mean(
        fluid.layers.cross_entropy(
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
    if program==None:
        program=fluid.default_main_program()
    func_parameters = {}
    for item in kwargs.items():
        if isinstance(item[1], str):
            func_parameters.setdefault(item[0],
                                       program.global_block().var(item[1]))
        else:
            func_parameters.setdefault(item[0], item[1])
    loss = loss_func(**func_parameters)
    return loss
