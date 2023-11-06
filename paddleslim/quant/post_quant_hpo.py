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
"""quant post with hyper params search"""

import os
import sys
import math
import time
from time import gmtime, strftime
import numpy as np
import shutil
import paddle
import logging
import argparse
import functools
import shutil
import glob
from scipy.stats import wasserstein_distance

from paddleslim.common import get_logger
from paddleslim.quant import quant_post

_logger = get_logger(__name__, level=logging.INFO)

SMAC_TMP_FILE_PATTERN = "smac3-output_"


def remove(path):
    """Remove files or directories matched by regex.
    Args:
        path(str): regular expressions to match the files and directories.
    """
    for p in glob.glob(path):
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


class QuantConfig(object):
    """quant config"""

    def __init__(self,
                 executor,
                 place,
                 float_infer_model_path,
                 quantize_model_path,
                 algo,
                 hist_percent,
                 bias_correct,
                 batch_size,
                 batch_num,
                 train_sample_generator=None,
                 eval_sample_generator=None,
                 train_dataloader=None,
                 eval_dataloader=None,
                 eval_function=None,
                 model_filename=None,
                 params_filename=None,
                 save_model_filename='model.pdmodel',
                 save_params_filename='model.pdiparams',
                 scope=None,
                 quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
                 is_full_quantize=False,
                 weight_bits=8,
                 activation_bits=8,
                 weight_quantize_type='channel_wise_abs_max',
                 optimize_model=False,
                 onnx_format=True,
                 is_use_cache_file=False,
                 cache_dir="./temp_post_training"):
        """QuantConfig init"""
        self.executor = executor
        self.place = place
        self.float_infer_model_path = float_infer_model_path.rstrip('/')
        self.quantize_model_path = quantize_model_path
        self.algo = algo,
        self.hist_percent = hist_percent,
        self.bias_correct = bias_correct,
        self.batch_size = batch_size,
        self.batch_num = batch_num,
        self.train_sample_generator = train_sample_generator
        self.eval_sample_generator = eval_sample_generator
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.eval_function = eval_function
        self.model_filename = model_filename
        self.params_filename = params_filename
        self.save_model_filename = save_model_filename
        self.save_params_filename = save_params_filename
        self.scope = scope
        self.quantizable_op_type = quantizable_op_type
        self.is_full_quantize = is_full_quantize
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.weight_quantize_type = weight_quantize_type
        self.optimize_model = optimize_model
        self.onnx_format = onnx_format
        self.is_use_cache_file = is_use_cache_file
        self.cache_dir = cache_dir


g_quant_config = None
g_quant_model_cache_path = "quant_model_tmp"


def emd_loss_init():
    global g_min_emd_loss
    g_min_emd_loss = float('inf')


emd_loss_init()


def make_feed_dict(feed_target_names, data):
    """construct feed dictionary"""
    feed_dict = {}
    if len(feed_target_names) == 1:
        feed_dict[feed_target_names[0]] = data
    else:
        for i in range(len(feed_target_names)):
            feed_dict[feed_target_names[i]] = data[i]
    return feed_dict


def standardization(data):
    """standardization numpy array"""
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    sigma = 1e-13 if sigma == 0. else sigma
    return (data - mu) / sigma


def cal_emd_lose(out_float_list, out_quant_list, out_len):
    """caculate earch move distance"""
    emd_sum = 0
    if out_len >= 3:
        for index in range(len(out_float_list)):
            emd_sum += wasserstein_distance(out_float_list[index],
                                            out_quant_list[index])
    else:
        out_float = np.concatenate(out_float_list)
        out_quant = np.concatenate(out_quant_list)
        emd_sum += wasserstein_distance(out_float, out_quant)
    emd_sum /= float(len(out_float_list))
    return emd_sum


def have_invalid_num(np_arr):
    """check have invalid number in numpy array"""
    have_invalid_num = False
    for val in np_arr:
        if math.isnan(val) or math.isinf(val):
            have_invalid_num = True
            break
    return have_invalid_num


def convert_model_out_2_nparr(model_out):
    """convert model output to numpy array"""
    if not isinstance(model_out, list):
        model_out = [model_out]
    out_list = []
    for out in model_out:
        out_list.append(np.array(out).reshape((1, )))

    out_nparr = np.concatenate(out_list)
    out_nparr = np.squeeze(out_nparr.flatten())
    return out_nparr


def eval_quant_model():
    """Eval quant model accuracy.
       Post quantization does not change the parameter value. Therefore, the closer the output distribution of the quantization model and the float model, the better the accuracy is maintained, 
       which has been verified in classification, detection, and nlp tasks. So the reward here is the earth mover distance between the output of the quantization model and the float model. 
       This distance measurement method is also verified on various tasks, and the stability is better than other distance measurement methods such as mse.
    """
    float_scope = paddle.static.Scope()
    quant_scope = paddle.static.Scope()
    with paddle.static.scope_guard(float_scope):
        [infer_prog_float, feed_target_names_float, fetch_targets_float] = \
            paddle.static.load_inference_model(path_prefix=g_quant_config.float_infer_model_path, \
            model_filename=g_quant_config.model_filename, \
            params_filename=g_quant_config.params_filename, \
            executor=g_quant_config.executor)

    with paddle.static.scope_guard(quant_scope):
        [infer_prog_quant, feed_target_names_quant, fetch_targets_quant] = \
            paddle.static.load_inference_model(path_prefix=g_quant_model_cache_path, \
            model_filename=g_quant_config.save_model_filename, \
            params_filename=g_quant_config.save_params_filename, \
            executor=g_quant_config.executor)

    out_float_list = []
    out_quant_list = []
    emd_sum = 0
    out_len_sum = 0
    valid_data_num = 0
    max_eval_data_num = 200
    if g_quant_config.eval_sample_generator is not None:
        feed_dict = False
        eval_dataloader = g_quant_config.eval_sample_generator
    else:
        feed_dict = True
        eval_dataloader = g_quant_config.eval_dataloader
    for i, data in enumerate(eval_dataloader()):
        with paddle.static.scope_guard(float_scope):
            out_float = g_quant_config.executor.run(infer_prog_float, \
                fetch_list=fetch_targets_float, feed=data if feed_dict else make_feed_dict(feed_target_names_float, data))
        with paddle.static.scope_guard(quant_scope):
            out_quant = g_quant_config.executor.run(infer_prog_quant, \
                fetch_list=fetch_targets_quant, feed=data if feed_dict else make_feed_dict(feed_target_names_quant, data))
        print(f"out_float: {out_float}")
        print(f"out_quant: {out_quant}")
        out_float = convert_model_out_2_nparr(out_float)
        out_quant = convert_model_out_2_nparr(out_quant)
        if len(out_float.shape) <= 0 or len(out_quant.shape) <= 0:
            continue

        min_len = min(out_float.shape[0], out_quant.shape[0])
        out_float = out_float[:min_len]
        out_quant = out_quant[:min_len]
        out_len_sum += min_len

        if have_invalid_num(out_float) or have_invalid_num(out_quant):
            continue

        try:
            if len(out_float) > 3:
                out_float = standardization(out_float)
                out_quant = standardization(out_quant)
        except:
            continue
        out_float_list.append(out_float)
        out_quant_list.append(out_quant)
        valid_data_num += 1

        if valid_data_num >= max_eval_data_num:
            break

    emd_sum = cal_emd_lose(out_float_list, out_quant_list,
                           out_len_sum / float(valid_data_num))
    _logger.info("output diff: {}".format(emd_sum))
    return float(emd_sum)


def quantize(cfg):
    """model quantize job"""
    algo = cfg["algo"] if 'algo' in cfg else g_quant_config.algo[0][0]
    if g_quant_config.hist_percent[0] is None:
        g_quant_config.hist_percent = [g_quant_config.hist_percent]
    hist_percent = cfg[
        "hist_percent"] if "hist_percent" in cfg else g_quant_config.hist_percent[
            0][0]
    bias_correct = cfg[
        "bias_correct"] if "bias_correct" in cfg else g_quant_config.bias_correct[
            0][0]
    batch_size = cfg[
        "batch_size"] if "batch_size" in cfg else g_quant_config.batch_size[0][
            0]
    batch_num = cfg[
        "batch_num"] if "batch_num" in cfg else g_quant_config.batch_num[0][0]
    weight_quantize_type = cfg[
        "weight_quantize_type"] if "weight_quantize_type" in cfg else g_quant_config.weight_quantize_type[
            0]

    quant_post( \
        executor=g_quant_config.executor, \
        scope=g_quant_config.scope, \
        model_dir=g_quant_config.float_infer_model_path, \
        quantize_model_path=g_quant_model_cache_path, \
        sample_generator=g_quant_config.train_sample_generator, \
        data_loader=g_quant_config.train_dataloader,
        model_filename=g_quant_config.model_filename, \
        params_filename=g_quant_config.params_filename, \
        save_model_filename=g_quant_config.save_model_filename, \
        save_params_filename=g_quant_config.save_params_filename, \
        quantizable_op_type=g_quant_config.quantizable_op_type, \
        activation_quantize_type="moving_average_abs_max", \
        weight_quantize_type=weight_quantize_type, \
        algo=algo, \
        hist_percent=hist_percent, \
        bias_correction=bias_correct, \
        batch_size=batch_size, \
        batch_nums=batch_num,
        onnx_format=g_quant_config.onnx_format)

    global g_min_emd_loss
    if g_quant_config.eval_function is None:
        emd_loss = eval_quant_model()
    else:
        ### if eval_function is not None, use eval function provided by user.
        float_scope = paddle.static.Scope()
        quant_scope = paddle.static.Scope()
        with paddle.static.scope_guard(float_scope):
            [float_inference_program, feed_target_names, fetch_targets]= paddle.static.load_inference_model( \
                    path_prefix=g_quant_config.float_infer_model_path, \
                    model_filename=g_quant_config.model_filename, params_filename=g_quant_config.params_filename,
                    executor=g_quant_config.executor)
            float_metric = g_quant_config.eval_function(
                g_quant_config.executor, float_inference_program,
                feed_target_names, fetch_targets)

        with paddle.static.scope_guard(quant_scope):
            [quant_inference_program, feed_target_names, fetch_targets] = paddle.static.load_inference_model( \
                    path_prefix=g_quant_model_cache_path, \
                    model_filename=g_quant_config.model_filename, params_filename=g_quant_config.params_filename,
                    executor=g_quant_config.executor)
            quant_metric = g_quant_config.eval_function(
                g_quant_config.executor, quant_inference_program,
                feed_target_names, fetch_targets)

        emd_loss = float(abs(float_metric - quant_metric)) / float_metric

    _logger.info("emd loss: {}".format(emd_loss))
    if emd_loss < g_min_emd_loss:
        g_min_emd_loss = emd_loss
        if os.path.exists(g_quant_config.quantize_model_path):
            shutil.rmtree(g_quant_config.quantize_model_path)
        os.system("cp -r {0} {1}".format(g_quant_model_cache_path,
                                         g_quant_config.quantize_model_path))
    return emd_loss


def quant_post_hpo(
        executor,
        place,
        model_dir,
        quantize_model_path,
        train_sample_generator=None,
        eval_sample_generator=None,
        train_dataloader=None,
        eval_dataloader=None,
        eval_function=None,
        model_filename=None,
        params_filename=None,
        save_model_filename='model,pdmodel',
        save_params_filename='model.pdiparams',
        scope=None,
        quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
        is_full_quantize=False,
        weight_bits=8,
        activation_bits=8,
        weight_quantize_type=['channel_wise_abs_max'],
        algo=["KL", "hist", "avg", "mse"],
        bias_correct=[True, False],
        hist_percent=[0.98, 0.999],  ### uniform sample in list.
        batch_size=[10, 30],  ### uniform sample in list.
        batch_num=[10, 30],  ### uniform sample in list.
        optimize_model=False,
        onnx_format=True,
        is_use_cache_file=False,
        cache_dir="./temp_post_training",
        runcount_limit=30):
    """
    The function utilizes static post training quantization method to
    quantize the fp32 model. It uses calibrate data to calculate the
    scale factor of quantized variables, and inserts fake quantization
    and dequantization operators to obtain the quantized model.

    Args:
        executor(paddle.static.Executor): The executor to load, run and save the
            quantized model.
        place(paddle.CPUPlace or paddle.CUDAPlace): This parameter represents
            the executor run on which device.
        model_dir(str): The path of fp32 model that will be quantized, and
            the model and params that saved by ``paddle.static.io.save_inference_model``
            are under the path.
        quantize_model_path(str): The path to save quantized model using api
            ``paddle.static.io.save_inference_model``.
        train_sample_generator(Python Generator): The sample generator provides
            calibrate data for DataLoader, and it only returns a sample every time.
        eval_sample_generator(Python Generator): The sample generator provides
            evalution data for DataLoader, and it only returns a sample every time.
        model_filename(str, optional): The name of model file. If parameters
            are saved in separate files, set it as 'None'. Default: 'None'.
        params_filename(str, optional): The name of params file.
                When all parameters are saved in a single file, set it
                as filename. If parameters are saved in separate files,
                set it as 'None'. Default : 'None'.
        save_model_filename(str): The name of model file to save the quantized inference program.  Default: 'model.pdmodel'.
        save_params_filename(str): The name of file to save all related parameters. Default: 'model.pdiparams'.
        scope(paddle.static.Scope, optional): The scope to run program, use it to load
                        and save variables. If scope is None, will use paddle.static.global_scope().
        quantizable_op_type(list[str], optional): The list of op types
                        that will be quantized. Default: ["conv2d", "depthwise_conv2d",
                        "mul"].
        is_full_quantize(bool): if True, apply quantization to all supported quantizable op type.
                        If False, only apply quantization to the input quantizable_op_type. Default is False.
        weight_bits(int, optional): quantization bit number for weights.
        activation_bits(int): quantization bit number for activation.
        weight_quantize_type(str): quantization type for weights,
                support 'abs_max' and 'channel_wise_abs_max'. Compared to 'abs_max',
                the model accuracy is usually higher when using 'channel_wise_abs_max'.
        optimize_model(bool, optional): If set optimize_model as True, it applies some
                passes to optimize the model before quantization. So far, the place of
                executor must be cpu it supports fusing batch_norm into convs.
        onnx_format(bool): Whether to export the quantized model with format of ONNX. Default is False.
        is_use_cache_file(bool): This param is deprecated.
        cache_dir(str): This param is deprecated.
        runcount_limit(int): max. number of model quantization.
    Returns:
        None
    """

    try:
        import smac
        assert smac.version == '1.4.0'
    except:
        _logger.warning(
            "smac==1.4.0 is required, please use \"pip install smac==1.4.0\".")
        os.system('python -m pip install smac==1.4.0')

    from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
        UniformFloatHyperparameter, UniformIntegerHyperparameter
    from smac.configspace import ConfigurationSpace
    from smac.facade.smac_hpo_facade import SMAC4HPO
    from smac.scenario.scenario import Scenario

    global g_quant_config
    g_quant_config = QuantConfig(
        executor, place, model_dir, quantize_model_path, algo, hist_percent,
        bias_correct, batch_size, batch_num, train_sample_generator,
        eval_sample_generator, train_dataloader, eval_dataloader, eval_function,
        model_filename, params_filename, save_model_filename,
        save_params_filename, scope, quantizable_op_type, is_full_quantize,
        weight_bits, activation_bits, weight_quantize_type, optimize_model,
        onnx_format, is_use_cache_file, cache_dir)
    cs = ConfigurationSpace()

    hyper_params = []

    if 'hist' in algo:
        hist_percent = UniformFloatHyperparameter(
            "hist_percent",
            hist_percent[0],
            hist_percent[1],
            default_value=hist_percent[0])
        hyper_params.append(hist_percent)

    if len(algo) > 1:
        algo = CategoricalHyperparameter("algo", algo, default_value=algo[0])
        hyper_params.append(algo)
    else:
        algo = algo[0]

    if len(bias_correct) > 1:
        bias_correct = CategoricalHyperparameter(
            "bias_correct", bias_correct, default_value=bias_correct[0])
        hyper_params.append(bias_correct)
    else:
        bias_correct = bias_correct[0]
    if len(weight_quantize_type) > 1:
        weight_quantize_type = CategoricalHyperparameter("weight_quantize_type", \
            weight_quantize_type, default_value=weight_quantize_type[0])
        hyper_params.append(weight_quantize_type)
    else:
        weight_quantize_type = weight_quantize_type[0]
    if len(batch_size) > 1:
        batch_size = UniformIntegerHyperparameter(
            "batch_size",
            batch_size[0],
            batch_size[1],
            default_value=batch_size[0])
        hyper_params.append(batch_size)
    else:
        batch_size = batch_size[0]

    if len(batch_num) > 1:
        batch_num = UniformIntegerHyperparameter(
            "batch_num", batch_num[0], batch_num[1], default_value=batch_num[0])
        hyper_params.append(batch_num)
    else:
        batch_num = batch_num[0]

    if len(hyper_params) == 0:
        quant_post( \
            executor=g_quant_config.executor, \
            scope=g_quant_config.scope, \
            model_dir=g_quant_config.float_infer_model_path, \
            quantize_model_path=g_quant_model_cache_path, \
            sample_generator=g_quant_config.train_sample_generator, \
            data_loader=g_quant_config.train_dataloader,
            model_filename=g_quant_config.model_filename, \
            params_filename=g_quant_config.params_filename, \
            save_model_filename=g_quant_config.save_model_filename, \
            save_params_filename=g_quant_config.save_params_filename, \
            quantizable_op_type=g_quant_config.quantizable_op_type, \
            activation_quantize_type="moving_average_abs_max", \
            weight_quantize_type=weight_quantize_type, \
            algo=algo, \
            hist_percent=hist_percent, \
            bias_correction=bias_correct, \
            batch_size=batch_size, \
            batch_nums=batch_num)
        shutil.rmtree(g_quant_model_cache_path)
        return

    cs.add_hyperparameters(hyper_params)

    s_datetime = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
    smac_output_dir = SMAC_TMP_FILE_PATTERN + s_datetime

    scenario = Scenario({
        "run_obj": "quality",  # we optimize quality (alternative runtime)
        "runcount-limit":
        runcount_limit,  # max. number of function evaluations; for this example set to a low number
        "cs": cs,  # configuration space
        "deterministic": "True",
        "limit_resources": "False",
        "memory_limit":
        4096,  # adapt this to reasonable value for your hardware
        "output_dir": smac_output_dir  # output_dir
    })
    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4HPO(
        scenario=scenario, rng=np.random.RandomState(42), tae_runner=quantize)

    # Example call of the function with default values
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = smac.get_tae_runner().run(cs.get_default_configuration(), 1)[1]
    _logger.info("Value for default configuration: %.8f" % def_value)

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = smac.get_tae_runner().run(incumbent, 1)[1]
    _logger.info("Optimized Value: %.8f" % inc_value)
    shutil.rmtree(g_quant_model_cache_path)
    remove(smac_output_dir)
    _logger.info("Quantization completed.")
