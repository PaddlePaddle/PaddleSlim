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

import os
import logging

import paddle
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization
from paddle.fluid.contrib.slim.quantization import WeightQuantization
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.contrib.slim.quantization.quanter import WEIGHT_QUANTIZATION_TYPES
from paddle.fluid.contrib.slim.quantization.quanter import WEIGHT_QUANTIZATION_TYPES_TENSORRT
from paddle.fluid.contrib.slim.quantization.quanter import ACTIVATION_QUANTIZATION_TYPES
from paddle.fluid.contrib.slim.quantization.quanter import ACTIVATION_QUANTIZATION_TYPES_TENSORRT
from paddle.fluid.contrib.slim.quantization.quanter import VALID_DTYPES
from paddle.fluid.contrib.slim.quantization.quanter import TENSORRT_OP_TYPES
from paddle.fluid.contrib.slim.quantization.quanter import VARS_MAPPING_TABLE
from paddle.fluid.contrib.slim.quantization.quanter import _quant_config_default
from paddle.fluid.contrib.slim.quantization.quanter import load_dict
from paddle.fluid.contrib.slim.quantization.quanter import save_dict
from paddle.fluid.contrib.slim.quantization.quanter import _parse_configs
from paddle.fluid.contrib.slim.quantization.quanter import quant_aware
from paddle.fluid.contrib.slim.quantization.quanter import convert


def quant_post_static(
        executor,
        model_dir,
        quantize_model_path,
        batch_generator=None,
        sample_generator=None,
        data_loader=None,
        model_filename=None,
        params_filename=None,
        save_model_filename='__model__',
        save_params_filename='__params__',
        batch_size=1,
        batch_nums=None,
        scope=None,
        algo='hist',
        round_type='round',
        hist_percent=0.9999,
        bias_correction=False,
        quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
        is_full_quantize=False,
        weight_bits=8,
        activation_bits=8,
        activation_quantize_type='range_abs_max',
        weight_quantize_type='channel_wise_abs_max',
        optimize_model=False,
        onnx_format=False,
        skip_tensor_list=None,
        is_use_cache_file=False,
        cache_dir="./temp_post_training"):
    """
    The function utilizes static post training quantization method to
    quantize the fp32 model. It uses calibrate data to calculate the
    scale factor of quantized variables, and inserts fake quantization
    and dequantization operators to obtain the quantized model.

    Args:
        executor(paddle.static.Executor): The executor to load, run and save the 
            quantized model.
        model_dir(str): The path of fp32 model that will be quantized, and 
            the model and params that saved by ``paddle.static.io.save_inference_model`` 
            are under the path.
        quantize_model_path(str): The path to save quantized model using api
            ``paddle.static.io.save_inference_model``.
        batch_generator(Python Generator): The batch generator provides 
                calibrate data for DataLoader, and it returns a batch every
                time. For sample_generator and batch_generator, only one
                can be set. Beisdes, batch_generator supports lod tensor.
        sample_generator(Python Generator): The sample generator provides 
            calibrate data for DataLoader, and it only returns a sample every time.
        data_loader(Python Generator, Paddle.io.DataLoader, optional): The
            Generator or Dataloader provides calibrate data, and it could
            return a batch every time.
        model_filename(str, optional): The name of model file. If parameters 
            are saved in separate files, set it as 'None'. Default: 'None'.
        params_filename(str, optional): The name of params file.
                When all parameters are saved in a single file, set it 
                as filename. If parameters are saved in separate files, 
                set it as 'None'. Default : 'None'.
        save_model_filename(str): The name of model file to save the quantized inference program.  Default: '__model__'.
        save_params_filename(str): The name of file to save all related parameters. 
                If it is set None, parameters will be saved in separate files. Default: '__params__'.
        batch_size(int, optional): The batch size of DataLoader, default is 1.
        batch_nums(int, optional): If batch_nums is not None, the number of calibrate 
                        data is 'batch_size*batch_nums'. If batch_nums is None, use all data
                        generated by sample_generator  as calibrate data.
        scope(paddle.static.Scope, optional): The scope to run program, use it to load 
                        and save variables. If scope is None, will use paddle.static.global_scope().
        algo(str, optional): If algo='KL', use KL-divergenc method to 
                        get the scale factor. If algo='hist', use the hist_percent of histogram 
                        to get the scale factor. If algo='mse', search for the best scale factor which
                        makes the mse loss minimal. Use one batch of data for mse is enough. If 
                        algo='avg', use the average of abs_max values  to get the scale factor. If 
                        algo='abs_max', use abs_max method to get the scale factor. Default: 'hist'.
        round_type(str, optional): The method of converting the quantized weights value
                        from float to int. Currently supports ['round', 'adaround'] methods.
                        Default is `round`, which is rounding nearest to the nearest whole number.
        hist_percent(float, optional): The percentile of histogram for algo hist.Default:0.9999.
        bias_correction(bool, optional): Bias correction method of https://arxiv.org/abs/1810.05723.
                        Default: False.
        quantizable_op_type(list[str], optional): The list of op types
                        that will be quantized. Default: ["conv2d", "depthwise_conv2d", 
                        "mul"].
        weight_bits(int, optional): quantization bit number for weights.
        activation_bits(int): quantization bit number for activation.
	activation_quantize_type(str): quantization type for activation,
                now support 'range_abs_max', 'moving_average_abs_max' and 'abs_max'.
                This parameter only specifies the fake ops in quantized model.
                If it is 'range_abs_max' or 'moving_average_abs_max', we save the scale
                obtained by post training quantization in fake ops. If it
                is 'abs_max', the scale will not be saved in fake ops.
        weight_quantize_type(str): quantization type for weights,
                support 'abs_max' and 'channel_wise_abs_max'. Compared to 'abs_max',
                the model accuracy is usually higher when using 'channel_wise_abs_max'.
        is_full_quantize(bool): if True, apply quantization to all supported quantizable op type.
                        If False, only apply quantization to the input quantizable_op_type. Default is False.
        optimize_model(bool, optional): If set optimize_model as True, it applies some 
                passes to optimize the model before quantization. So far, the place of
                executor must be cpu it supports fusing batch_norm into convs.
        onnx_format(bool): Whether to export the quantized model with format of ONNX. Default is False.
        skip_tensor_list(list): List of skip quant tensor name.
        is_use_cache_file(bool): This param is deprecated.
        cache_dir(str): This param is deprecated.
    
    Returns:
        None
    """
    try:
        post_training_quantization = PostTrainingQuantization(
            executor=executor,
            sample_generator=sample_generator,
            batch_generator=batch_generator,
            data_loader=data_loader,
            model_dir=model_dir,
            model_filename=model_filename,
            params_filename=params_filename,
            batch_size=batch_size,
            batch_nums=batch_nums,
            scope=scope,
            algo=algo,
            round_type=round_type,
            hist_percent=hist_percent,
            bias_correction=bias_correction,
            quantizable_op_type=quantizable_op_type,
            is_full_quantize=is_full_quantize,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            activation_quantize_type=activation_quantize_type,
            weight_quantize_type=weight_quantize_type,
            onnx_format=onnx_format,
            skip_tensor_list=skip_tensor_list,  # support in Paddle >= 2.3.1
            optimize_model=optimize_model)
    except:
        post_training_quantization = PostTrainingQuantization(
            executor=executor,
            sample_generator=sample_generator,
            batch_generator=batch_generator,
            data_loader=data_loader,
            model_dir=model_dir,
            model_filename=model_filename,
            params_filename=params_filename,
            batch_size=batch_size,
            batch_nums=batch_nums,
            scope=scope,
            algo=algo,
            round_type=round_type,
            hist_percent=hist_percent,
            bias_correction=bias_correction,
            quantizable_op_type=quantizable_op_type,
            is_full_quantize=is_full_quantize,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            activation_quantize_type=activation_quantize_type,
            weight_quantize_type=weight_quantize_type,
            onnx_format=onnx_format,
            optimize_model=optimize_model)

    post_training_quantization.quantize()
    post_training_quantization.save_quantized_model(
        quantize_model_path,
        model_filename=save_model_filename,
        params_filename=save_params_filename)


# We have changed the quant_post to quant_post_static.
# For compatibility, we keep quant_post api for now, and it will be
# deprecated in the future.
quant_post = quant_post_static


def quant_post_dynamic(model_dir,
                       save_model_dir,
                       model_filename=None,
                       params_filename=None,
                       save_model_filename=None,
                       save_params_filename=None,
                       quantizable_op_type=["conv2d", "mul"],
                       weight_bits=8,
                       generate_test_model=False):
    '''
    The function utilizes static post training quantization method to
    quantize the fp32 model. In details, it quantizes the weight of some
    ops from float32 to int8/16. For the quantized model, there are two
    kinds of calculation method in the reference stage. Firstly, the
    quantized weight will be dequantized to float32, and then apply the
    float32 calculation. Secondly, collect the quantized scales of the
    inputs, and then apply the int8 calculation.
        
    Args:
        model_dir(str): The path of the fp32 model that will be quantized,
                and the model and params files are under the path.
        save_model_dir(str): The path to save the quantized model.
        model_filename(str, optional): The name of file used to load the
                inference program. If it is None, the default filename
                '__model__' will be used. Default is 'None'.
        params_filename(str, optional): The name of file used to load all
                parameters. When all parameters were saved in a single
                binary file, set it as the real filename. If parameters
                were saved in separate files, set it as 'None'. Default is
                'None'.
        save_model_dir(str): The path used to save the quantized model.
        save_model_filename(str, optional): The name of file to 
                save the inference program. If it is None, the default 
                filename '__model__' will be used. Default is 'None'.
        save_params_filename(str, optional): The name of file to 
                save all parameters. If it is None, parameters were 
                saved in separate files. If it is not None, all 
                parameters were saved in a single binary file.
        quantizable_op_type(list[str], optional): The list of ops 
                that will be quantized, and the quantized ops should be
                contained in ["conv2d", "depthwise_conv2d", "mul"]. 
                Default is ["conv2d", "depthwise_conv2d", "mul"].
        weight_bits(int, optional): The bits for the quantized weight, 
                and it should be 8 or 16. Default is 8.
        generate_test_model(bool, optional): If set generate_test_model 
                as True, it saves a fake quantized model, in which the weights 
                are quantized and dequantized. We can use PaddlePaddle to load 
                the fake quantized model and test the accuracy on GPU or CPU.
    '''

    weight_quant = WeightQuantization(
        model_dir=model_dir,
        model_filename=model_filename,
        params_filename=params_filename)

    weight_quant.quantize_weight_to_int(
        save_model_dir=save_model_dir,
        save_model_filename=save_model_filename,
        save_params_filename=save_params_filename,
        quantizable_op_type=quantizable_op_type,
        weight_bits=weight_bits,
        generate_test_model=generate_test_model)


# We have changed the quant_post_only_weight to quant_post_dynamic.
# For compatibility, we keep quant_post_only_weight api for now,
# and it will be deprecated in the future.
quant_post_only_weight = quant_post_dynamic


def pact(x, name=None):
    helper = LayerHelper("pact", **locals())
    dtype = 'float32'
    init_thres = 20
    u_param_attr = paddle.fluid.ParamAttr(
        name=x.name + '_pact',
        initializer=paddle.fluid.initializer.ConstantInitializer(
            value=init_thres),
        regularizer=paddle.fluid.regularizer.L2Decay(0.0001),
        learning_rate=1)
    u_param = helper.create_parameter(attr=u_param_attr, shape=[1], dtype=dtype)
    x = paddle.fluid.layers.elementwise_sub(
        x,
        paddle.fluid.layers.relu(
            paddle.fluid.layers.elementwise_sub(x, u_param)))
    x = paddle.fluid.layers.elementwise_add(
        x,
        paddle.fluid.layers.relu(
            paddle.fluid.layers.elementwise_sub(-u_param, x)))

    return x


def get_pact_optimizer():
    return paddle.fluid.optimizer.MomentumOptimizer(0.0001, 0.9)
