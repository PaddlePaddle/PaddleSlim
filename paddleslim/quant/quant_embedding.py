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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import copy
import numpy as np
import math

import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid import core

from ..common import get_logger
_logger = get_logger(__name__, level=logging.INFO)

__all__ = ['quant_embedding']

_default_single_config = {
    "quantize_type": "abs_max",
    "quantize_bits": 8,
    "dtype": "int8"
}
SUPPORT_OP_TYPES = ['lookup_table', 'fused_embedding_seq_pool', 'pyramid_hash']
SUPPORT_QUANTIZE_TYPES = ['abs_max']
SUPPORT_QUANTIZE_BITS = [8]
SUPPORT_DTYPE = ['int8']

_default_config = {"quantize_op_types": SUPPORT_OP_TYPES, }


def _merge_config(old_config, new_config):
    """
    merge default config and user defined config

    Args:
        old_config(dict): the copy of default_config
        new_config(dict): the user defined config, 'params_name' must be set.
            When 'threshold' is not set, quant embedding without clip .
    """
    old_config.update(new_config)
    keys = old_config.keys()
    assert isinstance(old_config['quantize_op_types'], (str, list)), \
            'quantize_op_types can only be str or list[str]'
    if isinstance(old_config['quantize_op_types'], str):
        old_config['quantize_op_types'] = [old_config['quantize_op_types']]
    for op_type in old_config['quantize_op_types']:
        assert op_type in SUPPORT_OP_TYPES, \
                '{} is not supported, supported op types are {}'.format(
                        op_type, SUPPORT_OP_TYPES)
        if op_type not in keys:
            old_config[op_type] = _default_single_config
            continue
        else:
            assert isinstance(old_config[op_type], dict), \
                    "op type {}'s config must be dict"
            config_tmp = copy.deepcopy(_default_single_config)
            config_tmp.update(old_config[op_type])
            old_config[op_type] = config_tmp

        quantize_type = old_config[op_type]['quantize_type']
        assert isinstance(quantize_type, str), "quantize_type must be \
            str"

        assert quantize_type in SUPPORT_QUANTIZE_TYPES , "" \
            "quantize_type {} is not supported, now supported quantize type" \
            " are {}.".format(quantize_type, SUPPORT_QUANTIZE_TYPES)

        quantize_bits = old_config[op_type]['quantize_bits']
        assert isinstance(quantize_bits, int), "quantize_bits must be int"
        assert quantize_bits in SUPPORT_QUANTIZE_BITS , " quantize_bits {}" \
                " is not supported, now supported quantize bits are" \
                " {}. ".format(quantize_bits, SUPPORT_QUANTIZE_BITS)

        dtype = old_config[op_type]['dtype']
        assert isinstance(dtype, str), "dtype must be str"
        assert dtype in SUPPORT_DTYPE , " dtype {} is not "\
            "supported, now supported dtypes are {} ".format(dtype, SUPPORT_DTYPE)
        if 'threshold' in old_config[op_type].keys():
            assert isinstance(old_config[op_type]['threshold'], (float, int)), \
                    "threshold must be number."

    _logger.info("quant_embedding config {}".format(old_config))
    return old_config


def _get_var_tensor(scope, var_name):
    """
    get tensor array by name.
    Args:
        scope(fluid.Scope): scope to get var
        var_name(str): vatiable name
    Return:
        np.array
    """
    return np.array(scope.find_var(var_name).get_tensor())


def _get_scale_var_name(var_name):
    """
    get scale var name 
    """
    return var_name + '.scale'


def _get_quant_var_name(var_name):
    """
    get quantized var name
    """
    return var_name + '.int8'


def _get_dequant_var_name(var_name):
    """
    get dequantized var name
    """
    return var_name + '.dequantize'


def _restore_var(name, arr, scope, place):
    """
    restore quantized array to quantized var
    """
    tensor = scope.find_var(name).get_tensor()
    tensor.set(arr, place)


def _clear_var(var_name, scope):
    """
    free memory of var
    """
    tensor = scope.find_var(var_name).get_tensor()
    tensor._clear()


def _quant_embedding_abs_max(graph, scope, place, config, var_name,
                             embedding_node):
    """
    quantize embedding using abs_max

    Args:
        graph(IrGraph): graph that includes lookup_table op
        scope(fluid.Scope): scope
        place(fluid.CPUPlace or flud.CUDAPlace): place
        config(dict): config to quant
    """

    def _quant_abs_max(tensor_array, config):
        """
        quant array using abs_max op
        """
        bit_length = config['quantize_bits']
        scale = np.max(np.abs(tensor_array)).astype("float32")
        quanted_tensor = np.round(tensor_array / scale * (
            (1 << (bit_length - 1)) - 1))
        return scale, quanted_tensor.astype(config['dtype'])

    def _insert_dequant_abs_max_op(graph, scope, var_node, scale_node, config):
        """
        Insert dequantize_abs_max op in graph
        """
        assert var_node.is_var(), "{} is not a var".format(var_node.name())

        dequant_var_node = graph.create_var_node(
            name=_get_dequant_var_name(var_node.name()),
            var_type=var_node.type(),
            shape=var_node.shape(),
            var_dtype=core.VarDesc.VarType.FP32)
        scope.var(dequant_var_node.name())

        max_range = (1 << (config['quantize_bits'] - 1)) - 1
        output_ops = var_node.outputs
        dequant_op = graph.create_op_node(
            op_type='dequantize_abs_max',
            attrs={
                'max_range': float(max_range),
                'op_role': core.op_proto_and_checker_maker.OpRole.Forward
            },
            inputs={'X': var_node,
                    'Scale': scale_node},
            outputs={'Out': dequant_var_node})
        graph.link_to(var_node, dequant_op)
        graph.link_to(scale_node, dequant_op)
        graph.link_to(dequant_op, dequant_var_node)
        for node in output_ops:
            graph.update_input_link(var_node, dequant_var_node, node)

    def _clip_array(array, config):
        if 'threshold' in config.keys():
            threshold = config['threshold']
        else:
            abs_array = np.max(np.abs(array))
            print(abs_array)
            if abs_array < 1.0:
                return array
            threshold = np.percentile(np.abs(array), 99.99)
        return np.clip(array, -threshold, threshold)

    embedding_tensor = _get_var_tensor(scope, var_name)
    embedding_array = _clip_array(embedding_tensor, config)
    # get scale and quanted tensor
    scale, quanted_tensor = _quant_abs_max(embedding_array, config)

    #create params must to use create_persistable_node
    scale_var = graph.create_persistable_node(
        _get_scale_var_name(var_name),
        var_type=embedding_node.type(),
        shape=[1],
        var_dtype=core.VarDesc.VarType.FP32)
    quant_tensor_var = graph.create_persistable_node(
        _get_quant_var_name(var_name),
        var_type=embedding_node.type(),
        shape=embedding_node.shape(),
        var_dtype=core.VarDesc.VarType.INT8)
    # create var in scope
    scope.var(_get_quant_var_name(var_name))
    scope.var(_get_scale_var_name(var_name))
    #set var by tensor array or scale
    _restore_var(_get_quant_var_name(var_name), quanted_tensor, scope, place)
    _restore_var(_get_scale_var_name(var_name), np.array(scale), scope, place)

    # insert dequantize_abs_max op
    for op_node in embedding_node.outputs:
        graph.update_input_link(embedding_node, quant_tensor_var, op_node)
        out_name = op_node.output('Out')[0]
        var_node = graph._find_node_by_name(op_node.outputs, out_name)
        _insert_dequant_abs_max_op(graph, scope, var_node, scale_var, config)

    # free float embedding params memory
    _clear_var(embedding_node.name(), scope)
    graph.safe_remove_nodes(embedding_node)


def _remove_link(in_node, out_node):
    in_node.remove_output(out_node)
    out_node.remove_input(in_node)


def _split_embedding_seq_pool(graph, op):
    inputs = op.inputs
    outputs = op.outputs
    op_desc = op.node.op()
    combiner = op_desc.attr("combiner")
    padding_idx = op_desc.attr("padding_idx")
    is_sparse = op_desc.attr("is_sparse")
    ids = graph._find_node_by_name(inputs, op.input('Ids')[0])
    weight = graph._find_node_by_name(inputs, op.input('W')[0])
    out = outputs[0]
    lookup_out = graph.create_var_node(
        name=ids.name() + '.look_up_table.out',
        var_type=core.VarDesc.VarType.LOD_TENSOR,
        shape=[1],
        var_dtype=weight.dtype())
    lookup_table_op = graph.create_op_node(
        op_type='lookup_table',
        attrs={'is_sparse': is_sparse,
               'padding_idx': padding_idx},
        inputs={'W': weight,
                'Ids': ids},
        outputs={'Out': lookup_out})
    _remove_link(ids, op)
    _remove_link(weight, op)
    _remove_link(op, out)
    graph.link_to(ids, lookup_table_op)
    graph.link_to(weight, lookup_table_op)
    graph.link_to(lookup_table_op, lookup_out)
    max_index = graph.create_var_node(
        name=ids.name() + '.seq_pool_op.max_index',
        var_type=core.VarDesc.VarType.LOD_TENSOR,
        shape=[1],
        var_dtype=weight.dtype())

    seq_pool_op = graph.create_op_node(
        op_type='sequence_pool',
        inputs={'X': lookup_out},
        outputs={'Out': out,
                 'MaxIndex': max_index},
        attrs={'pooltype': combiner.upper(),
               'is_test': True})
    if combiner == 'max':
        max_index.stop_gradient = True
    graph.link_to(lookup_out, seq_pool_op)
    graph.link_to(seq_pool_op, out)
    graph.link_to(seq_pool_op, max_index)


def quant_embedding(program, place, config=None, scope=None):
    """quantize lookup_table op parameters

    Args:
        program(fluid.Program): infer program
        scope(fluid.Scope): Scope records the mapping between variable names and variables, similar to brackets in programming languages. Usually users can use `fluid.global_scope() <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_ . When ``None`` will use `fluid.global_scope() <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_. Default : ``None``.
        place(fluid.CPUPlace or fluid.CUDAPlace): This parameter represents the executor run on which device.
        config(dict): config to quantize. The keys are 'params_name', 'quantize_type', \
                'quantize_bits', 'dtype', 'threshold'. \
                ``quantize_type`` is  quantize type, supported types are ['abs_max'], default is "abs_max".
                ``quantize_bits`` supported bits are [8] and default is 8.
                ``dtype`` is quantize dtype, supported dtype are ['int8'], default is 'int8'.
                ``threshold`` is threshold to clip tensor before quant. When threshold is not set, \
                        tensor will not be clipped.

    Returns:
        None
    """
    config = {} if config is None else config
    config = _merge_config(copy.deepcopy(_default_config), config)
    scope = fluid.global_scope() if scope is None else scope

    graph = IrGraph(core.Graph(program.desc), for_test=True)
    quantize_params_map = {}
    all_op = graph.all_op_nodes()
    for op in all_op:
        if op.inputs == [] and op.outputs == []:
            continue
        op_type = op.name()
        if op_type in config['quantize_op_types']:
            weight_name = op.input('W')[0]
            if weight_name in quantize_params_map.values():
                continue
            embedding_node = graph._find_node_by_name(op.inputs,
                                                      op.input('W')[0])
            for op_node in embedding_node.outputs:
                if op_node.name() == 'fused_embedding_seq_pool':
                    _split_embedding_seq_pool(graph, op_node)
            _quant_embedding_abs_max(graph, scope, place, \
                    config[op_type], weight_name, embedding_node)
            quantize_params_map[weight_name] = _get_quant_var_name(weight_name)
    for op in all_op:
        if op.name() == 'fused_embedding_seq_pool':
            graph.safe_remove_nodes(op)

    return graph.to_program()
