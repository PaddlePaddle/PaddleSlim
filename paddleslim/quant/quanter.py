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
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization
from paddle.fluid.contrib.slim.quantization import AddQuantDequantPass
from paddle.fluid import core

from ..common import get_logger
_logger = get_logger(__name__, level=logging.INFO)

WEIGHT_QUANTIZATION_TYPES = [
    'abs_max', 'channel_wise_abs_max', 'range_abs_max',
    'moving_average_abs_max'
]
WEIGHT_QUANTIZATION_TYPES_TENSORRT = ['channel_wise_abs_max']

ACTIVATION_QUANTIZATION_TYPES = [
    'abs_max', 'range_abs_max', 'moving_average_abs_max'
]

ACTIVATION_QUANTIZATION_TYPES_TENSORRT = [
    'range_abs_max', 'moving_average_abs_max'
]

VALID_DTYPES = ['int8']
TRANSFORM_PASS_OP_TYPES = QuantizationTransformPass._supported_quantizable_op_type
QUANT_DEQUANT_PASS_OP_TYPES = AddQuantDequantPass._supported_quantizable_op_type
TENSORRT_OP_TYPES = [
    'mul', 'conv2d', 'pool2d', 'depthwise_conv2d', 'elementwise_add',
    'leaky_relu'
]

_quant_config_default = {
    # weight quantize type, default is 'channel_wise_abs_max'
    'weight_quantize_type': 'channel_wise_abs_max',
    # activation quantize type, default is 'moving_average_abs_max'
    'activation_quantize_type': 'moving_average_abs_max',
    # weight quantize bit num, default is 8
    'weight_bits': 8,
    # activation quantize bit num, default is 8
    'activation_bits': 8,
    # ops of name_scope in not_quant_pattern list, will not be quantized
    'not_quant_pattern': ['skip_quant'],
    # ops of type in quantize_op_types, will be quantized
    'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul'],
    # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
    'dtype': 'int8',
    # window size for 'range_abs_max' quantization. defaulf is 1000
    'window_size': 10000,
    # The decay coefficient of moving average, default is 0.9
    'moving_rate': 0.9,
    # if True, 'quantize_op_types' will be TENSORRT_OP_TYPES
    'for_tensorrt': False,
    # if True, 'quantoze_op_types' will be TRANSFORM_PASS_OP_TYPES + QUANT_DEQUANT_PASS_OP_TYPES 
    'is_full_quantize': False
}


def _parse_configs(user_config):
    """
    check if user's configs are valid.
    Args:
        user_config(dict): user's config.
    Return:
        configs(dict): final configs will be used.
    """

    configs = copy.deepcopy(_quant_config_default)
    configs.update(user_config)

    # check if configs is valid
    if configs['for_tensorrt']:
        weight_types = WEIGHT_QUANTIZATION_TYPES_TENSORRT
        activation_types = ACTIVATION_QUANTIZATION_TYPES_TENSORRT
        platform = 'TensorRT'
    else:
        weight_types = WEIGHT_QUANTIZATION_TYPES
        activation_types = WEIGHT_QUANTIZATION_TYPES
        platform = 'PaddleLite'
    assert configs['weight_quantize_type'] in weight_types, \
        "Unknown weight_quantize_type: {}. {} only supports {} ".format(configs['weight_quantize_type'],
                platform, weight_types)

    assert configs['activation_quantize_type'] in activation_types, \
        "Unknown activation_quantize_type: {}. {} only supports {}".format(configs['activation_quantize_type'],
                platform, activation_types)

    assert isinstance(configs['weight_bits'], int), \
        "weight_bits must be int value."

    assert (configs['weight_bits'] >= 1 and configs['weight_bits'] <= 16), \
        "weight_bits should be between 1 and 16."

    assert isinstance(configs['activation_bits'], int), \
        "activation_bits must be int value."

    assert (configs['activation_bits'] >= 1 and configs['activation_bits'] <= 16), \
        "activation_bits should be between 1 and 16."

    assert isinstance(configs['not_quant_pattern'], (list, str)), \
        "not_quant_pattern must be list or str"

    assert isinstance(configs['quantize_op_types'], list), \
        "quantize_op_types must be a list"

    if configs['for_tensorrt']:
        configs['quantize_op_types'] = TENSORRT_OP_TYPES
    elif configs['is_full_quantize']:
        configs[
            'quantize_op_types'] = TRANSFORM_PASS_OP_TYPES + QUANT_DEQUANT_PASS_OP_TYPES
    else:
        for op_type in configs['quantize_op_types']:
            assert (op_type in QUANT_DEQUANT_PASS_OP_TYPES) or (
                op_type in TRANSFORM_PASS_OP_TYPES), "{} is not support, \
                        now support op types are {}".format(
                    op_type,
                    TRANSFORM_PASS_OP_TYPES + QUANT_DEQUANT_PASS_OP_TYPES)

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
        program(fluid.Program): program to quant
        place(fluid.CPUPlace or fluid.CUDAPlace): CPU or CUDA device
        config(dict, optional): configs for quantization. if None, will use default config. Default is None.
        scope(fluid.Scope): the scope to store var, it should be program's scope. if None, will use fluid.global_scope().
            default is None.
        for_test(bool): if program is test program, set True when program is for test, False when program is for train. Default is False.
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

    transform_pass_ops = []
    quant_dequant_ops = []
    for op_type in config['quantize_op_types']:
        if op_type in TRANSFORM_PASS_OP_TYPES:
            transform_pass_ops.append(op_type)
        elif op_type in QUANT_DEQUANT_PASS_OP_TYPES:
            quant_dequant_ops.append(op_type)
    if len(transform_pass_ops) > 0:
        transform_pass = QuantizationTransformPass(
            scope=scope,
            place=place,
            weight_bits=config['weight_bits'],
            activation_bits=config['activation_bits'],
            activation_quantize_type=config['activation_quantize_type'],
            weight_quantize_type=config['weight_quantize_type'],
            window_size=config['window_size'],
            moving_rate=config['moving_rate'],
            quantizable_op_type=transform_pass_ops,
            skip_pattern=config['not_quant_pattern'])

        transform_pass.apply(main_graph)

    if len(quant_dequant_ops) > 0:
        quant_dequant_pass = AddQuantDequantPass(
            scope=scope,
            place=place,
            moving_rate=config['moving_rate'],
            quant_bits=config['activation_bits'],
            skip_pattern=config['not_quant_pattern'],
            quantizable_op_type=quant_dequant_ops)
        quant_dequant_pass.apply(main_graph)

    if for_test:
        quant_program = main_graph.to_program()
    else:
        quant_program = fluid.CompiledProgram(main_graph.graph)
    return quant_program


def quant_post(executor,
               model_dir,
               quantize_model_path,
               sample_generator,
               model_filename=None,
               params_filename=None,
               batch_size=16,
               batch_nums=None,
               scope=None,
               algo='KL',
               quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
               is_full_quantize=False,
               is_use_cache_file=False,
               cache_dir="./temp_post_training"):
    """
    The function utilizes post training quantization method to quantize the 
    fp32 model. It uses calibrate data to calculate the scale factor of 
    quantized variables, and inserts fake quant/dequant op to obtain the 
    quantized model.

    Args:
        executor(fluid.Executor): The executor to load, run and save the 
            quantized model.
        model_dir(str): The path of fp32 model that will be quantized, and 
            the model and params that saved by fluid.io.save_inference_model 
            are under the path.
        quantize_model_path(str): The path to save quantized model using api
            fluid.io.save_inference_model.
        sample_generator(Python Generator): The sample generator provides 
            calibrate data for DataLoader, and it only returns a sample every time.
        model_filename(str, optional): The name of model file. If parameters 
            are saved in separate files, set it as 'None'. Default is 'None'.
        params_filename(str, optional): The name of params file.
                When all parameters are saved in a single file, set it 
                as filename. If parameters are saved in separate files, 
                set it as 'None'. Default is 'None'.
        batch_size(int, optional): The batch size of DataLoader, default is 16.
        batch_nums(int, optional): If batch_nums is not None, the number of calibrate 
                        data is 'batch_size*batch_nums'. If batch_nums is None, use all data
                        generated by sample_generator  as calibrate data.
        scope(fluid.Scope, optional): The scope to run program, use it to load 
                        and save variables. If scope is None, will use fluid.global_scope().
        algo(str, optional): If algo=KL, use KL-divergenc method to 
                        get the more precise scale factor. If algo='direct', use 
                        abs_max method to get the scale factor. Default is 'KL'.
        quantizable_op_type(list[str], optional): The list of op types
                        that will be quantized. Default is ["conv2d", "depthwise_conv2d", 
                        "mul"].
        is_full_quantize(bool): if True, apply quantization to all supported quantizable op type.
                        If False, only apply quantization to the input quantizable_op_type. Default is False.
        is_use_cache_file(bool): If False, all temp data will be saved in memory. If True,
                                all temp data will be saved to disk. Defalut is False.
        cache_dir(str): When 'is_use_cache_file' is True, temp data will be save in 'cache_dir'. Default is './temp_post_training'.
    Returns:
        None
    """
    post_training_quantization = PostTrainingQuantization(
        executor=executor,
        sample_generator=sample_generator,
        model_dir=model_dir,
        model_filename=model_filename,
        params_filename=params_filename,
        batch_size=batch_size,
        batch_nums=batch_nums,
        scope=scope,
        algo=algo,
        quantizable_op_type=quantizable_op_type,
        is_full_quantize=is_full_quantize,
        is_use_cache_file=is_use_cache_file,
        cache_dir=cache_dir)
    post_training_quantization.quantize()
    post_training_quantization.save_quantized_model(quantize_model_path)


def convert(program, place, config=None, scope=None, save_int8=False):
    """
    change quantization ops order in program. return program that can used by Paddle-Lite.
    Args:
        program(fluid.Program): program that returned by quant_aware
        place(fluid.CPUPlace or fluid.CUDAPlace): CPU or CUDA device
        scope(fluid.Scope, optional):  the scope to store var, it should be program's scope. if None, will use fluid.global_scope().
            default is None.
        config(dict, optional): configs for convert. if set None, will use default config. Default is None.\
                It must be same with config that used in 'quant_aware'.
        save_int8: if return int8 freezed program. Int8 program can only be used to check size of model weights. \
                It cannot be used in Fluid or Paddle-Lite.
    Return:
        freezed_program(fluid.Program): freezed program which can be used for inference.
                       parameters is float32 type, but it's value in int8 range.
        freezed_program_int8(fluid.Program): freezed int8 program.
        when save_int8 is False, return freezed_program.
        when save_int8 is True, return freezed_program and freezed_program_int8
    """
    scope = fluid.global_scope() if not scope else scope

    if config is None:
        config = _quant_config_default
    else:
        assert isinstance(config, dict), "config must be dict"
        config = _parse_configs(config)
    _logger.info("convert config {}".format(config))

    test_graph = IrGraph(core.Graph(program.desc), for_test=True)
    support_op_types = []
    for op in config['quantize_op_types']:
        if op in QuantizationFreezePass._supported_quantizable_op_type:
            support_op_types.append(op)

    # Freeze the graph after training by adjusting the quantize
    # operators' order for the inference.
    freeze_pass = QuantizationFreezePass(
        scope=scope,
        place=place,
        weight_bits=config['weight_bits'],
        activation_bits=config['activation_bits'],
        weight_quantize_type=config['weight_quantize_type'],
        quantizable_op_type=support_op_types)
    freeze_pass.apply(test_graph)
    freezed_program = test_graph.to_program()

    if save_int8:
        convert_int8_pass = ConvertToInt8Pass(
            scope=fluid.global_scope(),
            place=place,
            quantizable_op_type=support_op_types)
        convert_int8_pass.apply(test_graph)
        freezed_program_int8 = test_graph.to_program()
        return freezed_program, freezed_program_int8
    else:
        return freezed_program
