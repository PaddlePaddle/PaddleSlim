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

import copy
import logging
import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization import QuantizationFreezePass
from paddle.fluid.contrib.slim.quantization import ConvertToInt8Pass
from paddle.fluid.contrib.slim.quantization import TransformForMobilePass
from paddle.fluid import core

from ..common import get_logger
_logger = get_logger(__name__, level=logging.INFO)

WEIGHT_QUANTIZATION_TYPES = [
    'abs_max', 'channel_wise_abs_max', 'range_abs_max',
    'moving_average_abs_max'
]
ACTIVATION_QUANTIZATION_TYPES = [
    'abs_max', 'range_abs_max', 'moving_average_abs_max'
]
VALID_DTYPES = ['int8']

_quant_config_default = {
    # weight quantize type, default is 'abs_max'
    'weight_quantize_type': 'abs_max',
    # activation quantize type, default is 'abs_max'
    'activation_quantize_type': 'abs_max',
    # weight quantize bit num, default is 8
    'weight_bits': 8,
    # activation quantize bit num, default is 8
    'activation_bits': 8,
    # ops of name_scope in not_quant_pattern , will not be quantized
    'not_quant_pattern': 'skip_quant',
    # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
    'dtype': 'int8',
    # window size for 'range_abs_max' quantization. defaulf is 10000
    'window_size': 10000,
    # The decay coefficient of moving average, default is 0.9
    'moving_rate': 0.9,
}


def _parse_configs(user_config):
    """
    check user configs is valid, and set default value if user not config.
    Args:
        user_config(dict):the config of user.
    Return:
        configs(dict): final configs will be used.
    """

    configs = copy.deepcopy(_quant_config_default)
    configs.update(user_config)

    # check configs is valid
    assert configs['weight_quantize_type'] in WEIGHT_QUANTIZATION_TYPES, \
        "Unknown weight_quantize_type: '%s'. It can only be " + " ".join(WEIGHT_QUANTIZATION_TYPES)

    assert configs['activation_quantize_type'] in ACTIVATION_QUANTIZATION_TYPES, \
        "Unknown activation_quantize_type: '%s'. It can only be " + " ".join(ACTIVATION_QUANTIZATION_TYPES)

    assert isinstance(configs['weight_bits'], int), \
        "weight_bits must be int value."

    assert (configs['weight_bits'] >= 1 and configs['weight_bits'] <= 16), \
        "weight_bits should be between 1 and 16."

    assert isinstance(configs['activation_bits'], int), \
        "activation_bits must be int value."

    assert (configs['activation_bits'] >= 1 and configs['activation_bits'] <= 16), \
        "activation_bits should be between 1 and 16."

    assert isinstance(configs['not_quant_pattern'], str), \
        "not_quant_pattern must be a list"

    assert isinstance(configs['dtype'], str), \
        "dtype must be a str."

    assert (configs['dtype'] in VALID_DTYPES), \
        "dtype can only be " + " ".join(VALID_DTYPES)

    assert isinstance(configs['window_size'], int), \
        "window_size must be int value, window size for 'range_abs_max' quantization, default is 10000."

    assert isinstance(configs['moving_rate'], float), \
        "moving_rate must be float value, The decay coefficient of moving average, default is 0.9."

    return configs


def quant_aware(program, place, config=None, scope=None, for_test=False):
    """
    add trainable quantization ops in program.
    Args:
        program(fluid.Program): program
        scope(fluid.Scope): the scope to store var, it's should be the value of program's scope, usually it's fluid.global_scope().
        place(fluid.CPUPlace or fluid.CUDAPlace): place
        config(dict): configs for quantization, default values are in quant_config_default dict.
        for_test: if program is test program, for_test should be set True, else False.
    Return:
        fluid.Program: user can finetune this quantization program to enhance the accuracy.
    """

    scope = fluid.global_scope() if not scope else scope
    if config is None:
        config = _quant_config_default
    else:
        assert isinstance(config, dict), "config must be dict"
        config = _parse_configs(config)
        _logger.info("quant_aware config {}".format(config))

    main_graph = IrGraph(core.Graph(program.desc), for_test=for_test)

    transform_pass = QuantizationTransformPass(
        scope=scope,
        place=place,
        weight_bits=config['weight_bits'],
        activation_bits=config['activation_bits'],
        activation_quantize_type=config['activation_quantize_type'],
        weight_quantize_type=config['weight_quantize_type'],
        window_size=config['window_size'],
        moving_rate=config['moving_rate'],
        skip_pattern=config['not_quant_pattern'])

    transform_pass.apply(main_graph)

    if for_test:
        quant_program = main_graph.to_program()
    else:
        quant_program = fluid.CompiledProgram(main_graph.graph)
    return quant_program


def convert(program, place, config=None, scope=None, save_int8=False):
    """
    add quantization ops in program. the program returned is not trainable.
    Args:
        program(fluid.Program): program
        scope(fluid.Scope): the scope to store var, when is None will use fluid.global_scope()
        place(fluid.CPUPlace or fluid.CUDAPlace): place
        config(dict): configs for quantization, default values are in quant_config_default dict.
        save_int8: is export int8 freezed program.
    Return:
        fluid.Program: freezed program which can be used for inference.
                       parameters is float32 type, but it's value in int8 range.
        fluid.Program: freezed int8 program which can be used for inference.
                       if save_int8 is False, this value is None.
    """
    scope = fluid.global_scope() if not scope else scope

    if config is None:
        config = _quant_config_default
    else:
        assert isinstance(config, dict), "config must be dict"
        config = _parse_configs(config)
    _logger.info("convert config {}".format(config))

    test_graph = IrGraph(core.Graph(program.desc), for_test=True)

    # Freeze the graph after training by adjusting the quantize
    # operators' order for the inference.
    freeze_pass = QuantizationFreezePass(
        scope=scope,
        place=place,
        weight_bits=config['weight_bits'],
        activation_bits=config['activation_bits'],
        weight_quantize_type=config['weight_quantize_type'])
    freeze_pass.apply(test_graph)
    freezed_program = test_graph.to_program()

    if save_int8:
        convert_int8_pass = ConvertToInt8Pass(
            scope=fluid.global_scope(), place=place)
        convert_int8_pass.apply(test_graph)
        freezed_program_int8 = test_graph.to_program()
        return freezed_program, freezed_program_int8
    else:
        return freezed_program
