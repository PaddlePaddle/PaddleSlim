#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import six
from paddle.fluid.framework import Parameter
from paddle.fluid import unique_name
from paddle.fluid import core
from ..core import GraphWrapper

__all__ = ['recover_inference_program']


def _remove_fetch_node(program):
    """remove fetch node in program"""
    for block in program.blocks:
        removed = 0
        ops = list(block.ops)
        for op in ops:
            if op.type == "fetch":
                idx = ops.index(op)
                block._remove_op(idx - removed)
                removed += 1


def _recover_reserve_space_with_bn(program):
    """Add the outputs which is only used for training and not saved in
       inference program."""
    for block_idx in six.moves.range(program.num_blocks):
        block = program.block(block_idx)
        for op in block.ops:
            if op.type == "batch_norm":
                if "ReserveSpace" not in op.output_names or len(
                        op.output("ReserveSpace")) == 0:
                    reserve_space = block.create_var(
                        name=unique_name.generate_with_ignorable_key(".".join(
                            ["reserve_space", 'tmp'])),
                        dtype=block.var(op.input("X")[0]).dtype,
                        type=core.VarDesc.VarType.LOD_TENSOR,
                        persistable=False,
                        stop_gradient=True)
                    op.desc.set_output("ReserveSpace", [reserve_space.name])
    return program


def _recover_param_attr(program):
    """recover parameters attribute.
       Params in infermodel are stored in the form of variable, which can not be trained."""
    all_weights = [param for param in program.list_vars() \
        if param.persistable is True and param.name != 'feed' and param.name != 'fetch']
    for w in all_weights:
        new_w = Parameter(
            block=program.block(0),
            shape=w.shape,
            dtype=w.dtype,
            type=w.type,
            name=w.name)
        new_w.set_value(w.get_value())
        program.block(0).vars[w.name] = new_w
    return program


def recover_inference_program(inference_program):
    """  recover inference program to train program which can be trained. """
    _remove_fetch_node(inference_program)
    inference_program = _recover_param_attr(inference_program)
    inference_program = _recover_reserve_space_with_bn(inference_program)
    for var in inference_program.list_vars():
        var.stop_gradient = False

    for op in inference_program.global_block().ops:
        op._set_attr("is_test", False)

    return inference_program
