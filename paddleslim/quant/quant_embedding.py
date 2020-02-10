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

import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid import core

#_logger = logging.basicConfig(level=logging.DEBUG)

__all__ = ['quant_embedding']

default_config = {
    "quantize_type": "abs_max",
    "quantize_bits": 8,
    "dtype": "int8"
}

support_quantize_types = ['abs_max']
support_quantize_bits = [8]
support_dtype = ['int8']


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
    assert 'params_name' in keys, "params_name must be set"

    quantize_type = old_config['quantize_type']
    assert isinstance(quantize_type, str), "quantize_type must be \
            str"

    assert quantize_type in support_quantize_types, " \
            quantize_type {} is not supported, now supported quantize type \
            are {}.".format(quantize_type, support_quantize_types)

    quantize_bits = old_config['quantize_bits']
    assert isinstance(quantize_bits, int), "quantize_bits must be int"
    assert quantize_bits in support_quantize_bits, " quantize_bits {} \
                is not supported, now supported quantize bits are \
                {}. ".format(quantize_bits, support_quantize_bits)

    dtype = old_config['dtype']
    assert isinstance(dtype, str), "dtype must be str"
    assert dtype in support_dtype, " dtype {} is not \
            supported, now supported dtypes are {} \
                ".format(dtype, support_dtype)
    if 'threshold' in keys:
        assert isinstance(old_config['threshold'], (float, int)), "threshold \
                must be number."

    print("quant_embedding config {}".format(old_config))
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


def _clip_tensor(tensor_array, threshold):
    """
    when 'threshold' is set, clip tensor by 'threshold' and '-threshold'
    Args:
        tensor_array(np.array): array to clip
        config(dict): config dict
    """
    tensor_array[tensor_array > threshold] = threshold
    tensor_array[tensor_array < -threshold] = -threshold
    return tensor_array


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


def _quant_embedding_abs_max(graph, scope, place, config):
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

    all_var_nodes = graph.all_var_nodes()
    var_name = config['params_name']
    # find embedding var node by 'params_name'
    embedding_node = graph._find_node_by_name(all_var_nodes, var_name)
    embedding_tensor = _get_var_tensor(scope, var_name)
    if 'threshold' in config.keys():
        embedding_tensor = _clip_tensor(embedding_tensor, config['threshold'])

    # get scale and quanted tensor
    scale, quanted_tensor = _quant_abs_max(embedding_tensor, config)

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
        if op_node.name() == 'lookup_table':
            graph.update_input_link(embedding_node, quant_tensor_var, op_node)
            var_node = op_node.outputs[0]
            _insert_dequant_abs_max_op(graph, scope, var_node, scale_var,
                                       config)

    # free float embedding params memory
    _clear_var(embedding_node.name(), scope)
    graph.safe_remove_nodes(embedding_node)


def quant_embedding(program, place, config, scope=None):
    """quantize lookup_table op parameters

    Args:
        program(fluid.Program): infer program
        scope(fluid.Scope): Scope records the mapping between variable names and variables, similar to brackets in programming languages. Usually users can use `fluid.global_scope() <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_ . When ``None`` will use `fluid.global_scope() <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_. Default : ``None``.
        place(fluid.CPUPlace or fluid.CUDAPlace): This parameter represents the executor run on which device.
        config(dict): config to quantize. The keys are 'params_name', 'quantize_type', \
                'quantize_bits', 'dtype', 'threshold'. \
                ``params_name`` is parameter name to quantize, must be set.
                ``quantize_type`` is  quantize type, supported types are ['abs_max'], default is "abs_max".
                ``quantize_bits`` supported bits are [8] and default is 8.
                ``dtype`` is quantize dtype, supported dtype are ['int8'], default is 'int8'.
                ``threshold`` is threshold to clip tensor before quant. When threshold is not set, \
                        tensor will not be clipped.

    Returns:
        None
    """
    assert isinstance(config, dict), "config must be dict"
    config = _merge_config(copy.deepcopy(default_config), config)
    scope = fluid.global_scope() if scope is None else scope

    graph = IrGraph(core.Graph(program.desc), for_test=True)
    if config['quantize_type'] == 'abs_max':
        _quant_embedding_abs_max(graph, scope, place, config)

    return graph.to_program()
