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
from ..common.recover_program import _recover_outputs_attr
from paddleslim.core import GraphWrapper
import paddle.nn.functional as F


def _find_var_from_program(program, var_name):
    for block in program.blocks:
        if block.has_var(var_name):
            return block.var(var_name)
    raise ValueError("var {} not in this program".format(var_name))


def _except_feed_fetch(var_name, merge_feed):
    if var_name != 'fetch' and (not merge_feed or var_name != 'feed'):
        return True
    return False


def _is_same_block(block1, block2):
    if len(block1.ops) != len(block2.ops):
        return False

    for op1, op2 in zip(block1.ops, block2.ops):
        if op1.type != op2.type:
            return False

    return True


def merge(teacher_program,
          student_program,
          data_name_map,
          place,
          scope=None,
          teacher_scope=None,
          name_prefix='teacher_',
          merge_feed=True):
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
        merge_feed(bool): Wheather to merge feed op when merge program. Default: True.

    Returns:
        None
    """
    if scope == None:
        scope = paddle.static.global_scope()
    if teacher_scope == None:
        teacher_scope = scope

    teacher_program = teacher_program.clone(for_test=True)
    teacher_program = _recover_outputs_attr(teacher_program)

    is_same_model = True
    if len(student_program.blocks) == len(teacher_program.blocks):
        for block in teacher_program.blocks:
            if not _is_same_block(block, student_program.block(block.idx)):
                is_same_model = False
                break
    else:
        is_same_model = False

    if is_same_model:
        for block in student_program.blocks:
            for op in block.ops:
                if op.type == 'while':
                    tmp_var = []
                    for _var_name in op.input('X'):
                        tmp_var.append('teacher_' + _var_name)
                    tmp_var.extend(op.input('X'))
                    op.desc.set_input("X", tmp_var)

    for block in teacher_program.blocks:
        for teacher_var in list(block.vars.values()):
            skip_rename = False
            if _except_feed_fetch(teacher_var.name, merge_feed):
                if teacher_var.name in data_name_map.keys():
                    new_name = data_name_map[teacher_var.name]
                    if new_name == teacher_var.name:
                        skip_rename = True
                else:
                    new_name = name_prefix + teacher_var.name
                if not skip_rename:
                    # scope var rename
                    old_var = teacher_scope.var(teacher_var.name).get_tensor()
                    renamed_var = scope.var(new_name).get_tensor()
                    renamed_var.set(np.array(old_var), place)

                    # program var rename
                    renamed_var = block._rename_var(teacher_var.name, new_name)

        ### input and output of the sub_block need to rename specially.
        for op in block.ops:
            for iname in op.input_names:
                for in_var_name in op.input(iname):
                    if _except_feed_fetch(
                            in_var_name,
                            merge_feed) and not block.has_var(in_var_name):
                        if in_var_name in data_name_map.keys():
                            new_name = data_name_map[in_var_name]
                            if new_name != in_var_name:
                                op._rename_input(in_var_name,
                                                 name_prefix + in_var_name)
                        else:
                            op._rename_input(in_var_name,
                                             name_prefix + in_var_name)

            for oname in op.output_names:
                for out_var_name in op.output(oname):
                    if _except_feed_fetch(
                            out_var_name,
                            merge_feed) and not block.has_var(out_var_name):
                        if out_var_name in data_name_map.keys():
                            new_name = data_name_map[out_var_name]
                            if new_name != out_var_name:
                                op._rename_output(out_var_name,
                                                  name_prefix + out_var_name)
                        else:
                            op._rename_output(out_var_name,
                                              name_prefix + out_var_name)

    for block in teacher_program.blocks:
        for teacher_var in list(block.vars.values()):
            if teacher_var.name != 'fetch' and (not merge_feed or
                                                teacher_var.name != 'feed'):
                # student program add var
                if len(student_program.blocks) > 1 and is_same_model:
                    new_var = student_program.block(block.idx)._clone_variable(
                        teacher_var, force_persistable=False)
                else:
                    new_var = student_program.global_block()._clone_variable(
                        teacher_var, force_persistable=False)
                new_var.stop_gradient = True

    for block in reversed(teacher_program.blocks):
        for op_idx, op in enumerate(block.ops):
            if (not merge_feed or op.type != 'feed') and op.type != 'fetch':
                inputs = {}
                outputs = {}
                attrs = {}
                for input_name in op.input_names:
                    inputs[input_name] = []
                    for in_var_name in op.input(input_name):
                        inputs[input_name].append(
                            block._find_var_recursive(in_var_name))

                for output_name in op.output_names:
                    outputs[output_name] = []
                    for out_var_name in op.output(output_name):
                        outputs[output_name].append(
                            block._find_var_recursive(out_var_name))

                for attr_name in op.attr_names:
                    if attr_name == 'sub_block':
                        attrs[attr_name] = student_program.block(
                            op._block_attr("sub_block").idx)
                    else:
                        attrs[attr_name] = op.attr(attr_name)
                if len(student_program.blocks) > 1 and is_same_model:
                    student_program.block(op.block.idx)._insert_op(
                        2 * op_idx,
                        type=op.type,
                        inputs=inputs,
                        outputs=outputs,
                        attrs=attrs)
                else:
                    student_program.global_block().append_op(
                        type=op.type,
                        inputs=inputs,
                        outputs=outputs,
                        attrs=attrs)

        student_program._sync_with_cpp()

    student_graph = GraphWrapper(student_program)
    for op in student_graph.ops():
        belongsto_teacher = False
        for inp in op.all_inputs():
            if 'teacher' in inp.name():
                belongsto_teacher = True
                break
        if belongsto_teacher:
            op._op._set_attr("skip_quant", True)


def fsp(teacher_var1_name,
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
    teacher_var1 = _find_var_from_program(program, teacher_var1_name)
    teacher_var2 = _find_var_from_program(program, teacher_var2_name)
    student_var1 = _find_var_from_program(program, student_var1_name)
    student_var2 = _find_var_from_program(program, student_var2_name)
    teacher_fsp_matrix = paddle.fluid.layers.fsp_matrix(teacher_var1,
                                                        teacher_var2)
    student_fsp_matrix = paddle.fluid.layers.fsp_matrix(student_var1,
                                                        student_var2)
    fsp_loss = paddle.mean(
        paddle.nn.functional.square_error_cost(student_fsp_matrix,
                                               teacher_fsp_matrix))
    return fsp_loss


def l2(teacher_var_name, student_var_name, program=None):
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
    student_var = _find_var_from_program(program, student_var_name)
    teacher_var = _find_var_from_program(program, teacher_var_name)
    l2_loss = paddle.mean(
        paddle.nn.functional.square_error_cost(student_var, teacher_var))
    return l2_loss


def soft_label(teacher_var_name,
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
    student_var = _find_var_from_program(program, student_var_name)
    teacher_var = _find_var_from_program(program, teacher_var_name)
    teacher_var.stop_gradient = True

    student_var = paddle.nn.functional.softmax(student_var /
                                               student_temperature)
    teacher_var = paddle.nn.functional.softmax(teacher_var /
                                               teacher_temperature)
    soft_label_loss = paddle.mean(
        paddle.nn.functional.cross_entropy(
            input=student_var,
            label=teacher_var,
            soft_label=True,
            use_softmax=False))
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
                                       _find_var_from_program(program, item[1]))
        else:
            func_parameters.setdefault(item[0], item[1])
    loss = loss_func(**func_parameters)
    return loss


def _top_mask(x):
    top_value, top_index = paddle.topk(x, 1)
    return paddle.cast(x == top_value, "int32")


def _cal_tc_nc_pred(x, top_mask):
    """Calculate the predictions of target class and non-target class.
    The predictions of target class is a binary distribution.
    And after removing the target class, the softmax on the remaining
    parts produces the non-target predictions.
    """
    pred = paddle.nn.functional.softmax(x)
    fp_mask = paddle.cast(top_mask, "float32")
    top_value = paddle.sum(fp_mask * pred, axis=1, keepdim=True)
    tc_pred = paddle.concat([top_value, 1 - top_value], axis=1)
    tmp = paddle.assign(x)
    tmp = tmp + (-100000 * top_mask)
    nc_pred = paddle.nn.functional.softmax(tmp)
    return tc_pred, nc_pred


def _dkd_loss(student_logits,
              teacher_logits,
              temperature=1.0,
              alpha=1.0,
              beta=1.0):
    mask = _top_mask(teacher_logits)
    print(f"mask: {mask.shape}")
    print(
        f"student_logits: {student_logits.shape}; teacher_logits: {teacher_logits.shape}"
    )
    s_tc_pred, s_nc_pred = _cal_tc_nc_pred(student_logits / temperature, mask)
    t_tc_pred, t_nc_pred = _cal_tc_nc_pred(teacher_logits / temperature, mask)
    tc_loss = paddle.nn.functional.kl_div(
        s_tc_pred, t_tc_pred, reduction='mean')
    nc_loss = paddle.nn.functional.kl_div(
        s_nc_pred, t_nc_pred, reduction='mean')
    loss = alpha * tc_loss + beta * nc_loss
    return loss * temperature**2


def dkd(teacher_var_name,
        student_var_name,
        program=None,
        temperature=1.0,
        alpha=1.0,
        beta=1.0):
    """Combine variables from student model and teacher model
    by Decoupled Knowledge Distillation loss (aka. dkd-loss).
    Reference: https://github.com/megvii-research/mdistiller
    Args:
        teacher_var_name(str): The name of teacher_var.
        student_var_name(str): The name of student_var.
        program(Program): The input distiller program. If not specified,
                          the default program will be used. Default: None
        temperature(float): Temperature used to divide
            teacher_feature_map before softmax. Default: 1.0
        alpha(float): The weight of target class loss. Default: 1.0
        beta(float): The weight of none-target class loss. Default: 1.0

    Returns: 
        Variable: dkd distiller loss.
    """
    if program == None:
        program = paddle.static.default_main_program()
    student_var = _find_var_from_program(program, student_var_name)
    teacher_var = _find_var_from_program(program, teacher_var_name)
    return _dkd_loss(
        student_var,
        teacher_var,
        temperature=temperature,
        alpha=alpha,
        beta=beta)


def skd(teacher_var_name, student_var_name, program=None, multiplier=None):
    """Combine variables from student model and teacher model 
    by Spherical Knowledge Distillation loss (aka. skd-loss).
    Reference: https://github.com/forjiuzhou/Spherical-Knowledge-Distillation
    Args:
        teacher_var_name(str): The name of teacher_var.
        student_var_name(str): The name of student_var.
        program(Program): The input distiller program. If not specified,
                          the default program will be used. Default: None
        multiplier(float): The multiplier to recover its norm to the original 
        level. When it's None, the appropriate multiplier can be computed by 
        teacher's logits with paddle.std(output_t, axis=1). Default: None.

    Returns:
        Variable: skd distiller loss.
    """
    if program == None:
        program = paddle.static.default_main_program()

    student_var = program.global_block().var(student_var_name)
    teacher_var = program.global_block().var(teacher_var_name)
    teacher_var.stop_gradient = True

    if multiplier is None:
        multiplier = paddle.std(teacher_var, axis=1, keepdim=True)

    logits_student = F.layer_norm(
        student_var,
        student_var.shape[1:],
        weight=None,
        bias=None,
        epsilon=1e-7) * multiplier
    logits_teacher = F.layer_norm(
        teacher_var,
        teacher_var.shape[1:],
        weight=None,
        bias=None,
        epsilon=1e-7) * multiplier

    student_out = F.softmax(logits_student, axis=1)
    teacher_out = F.softmax(logits_teacher, axis=1)
    skd_loss = paddle.mean(
        F.cross_entropy(
            input=student_out,
            label=teacher_out,
            soft_label=True,
            use_softmax=False))
    return skd_loss
