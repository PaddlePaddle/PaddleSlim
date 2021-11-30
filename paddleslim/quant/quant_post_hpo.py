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
"""quant post with hyper params search"""

import os
import cv2
import sys
import shutil
import logging
import paddle
import argparse
import functools
import math
import time
import numpy as np
import paddle.fluid as fluid
from scipy.stats import wasserstein_distance

# smac
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

from paddleslim.common import get_logger
from paddleslim.quant import quant_post

class QuantConfig:
    """quant config"""
    def __init__(self, 
        executor,
        place,
        float_infer_model_path,
        quantize_model_path,
        train_sample_generator=None,
        eval_sample_generator=None,
        model_filename=None,
        params_filename=None,
        save_model_filename='__model__',
        save_params_filename='__params__',
        scope=None,
        quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
        is_full_quantize=False,
        weight_bits=8,
        activation_bits=8,
        weight_quantize_type='channel_wise_abs_max',
        optimize_model=False,
        is_use_cache_file=False,
        cache_dir="./temp_post_training"):
        """init func"""
        self.executor=executor
        self.place=place
        self.float_infer_model_path = float_infer_model_path
        self.quantize_model_path=quantize_model_path
        self.train_sample_generator=train_sample_generator
        self.eval_sample_generator=eval_sample_generator
        self.model_filename=model_filename
        self.params_filename=params_filename
        self.save_model_filename=save_model_filename
        self.save_params_filename=save_params_filename
        self.scope=scope
        self.quantizable_op_type=quantizable_op_type
        self.is_full_quantize=is_full_quantize
        self.weight_bits=weight_bits
        self.activation_bits=activation_bits
        self.weight_quantize_type=weight_quantize_type
        self.optimize_model=optimize_model
        self.is_use_cache_file=is_use_cache_file
        self.cache_dir=cache_dir
g_quant_config = None
g_min_emd_loss = float('inf')

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
    return (data - mu) / sigma

def cal_emd_lose(out_float_list, out_quant_list, out_len):
    """caculate earch move distance"""
    emd_sum = 0
    if out_len >= 3:
        for index in range(len(out_float_list)):
            emd_sum += wasserstein_distance(out_float_list[index], out_quant_list[index])
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
        out_list.append(np.array(out))

    out_nparr = np.concatenate(out_list)
    out_nparr = np.squeeze(out_nparr.flatten())
    return out_nparr

def eval_quant_model():
    """eval quant model accuracy"""
    float_scope = paddle.static.Scope()
    quant_scope = paddle.static.Scope()
    with paddle.static.scope_guard(float_scope):
        [infer_prog_float, feed_target_names_float, fetch_targets_float] = \
            fluid.io.load_inference_model(dirname=args.model_path, \
            model_filename=args.input_model_filename, \
            params_filename=args.input_params_filename, \
            executor=exe)

    with paddle.static.scope_guard(quant_scope):
        [infer_prog_quant, feed_target_names_quant, fetch_targets_quant] = \
            fluid.io.load_inference_model(dirname=args.save_path, \
            model_filename=args.save_model_filename, \
            params_filename=args.save_params_filename, \
            executor=exe)

    emd_sum = 0
    out_len_sum = 0
    max_eval_data_num = 200
    for i, data in enumerate(g_quant_config.eval_sample_generator()):
        #print('data shape: ', data.shape)
        with paddle.static.scope_guard(float_scope):
            out_float = exe.run(infer_prog_float, \
                fetch_list=fetch_targets_float, feed=make_feed_dict(feed_target_names_float, data)) 
        with paddle.static.scope_guard(quant_scope):
            out_quant = exe.run(infer_prog_quant, \
                fetch_list=fetch_targets_quant, feed=make_feed_dict(feed_target_names_quant, data)) 

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
            out_float = standardization(out_float)
            out_quant = standardization(out_quant)
        except:
            continue
        out_float_list.append(out_float)
        out_quant_list.append(out_quant)
        valid_data_num += 1

        if valid_data_num >= max_eval_data_num:
            break

    emd_sum = cal_emd_lose(out_float_list, out_quant_list, out_len_sum / float(valid_data_num))
    print("output diff:", emd_sum)
    return float(emd_sum)

def quantize(cfg):
    """model quantize job""" 
    algo = cfg["algo"]
    hist_percent = cfg["hist_percent"]
    bias_correct = cfg["bias_correct"]
    activation_quantize_method = cfg["activation_quantize_method"]
    batch_size = cfg["batch_size"]
    batch_num = cfg["batch_num"]

    quant_model_cache = "quant_model_tmp"
    quant_post( \
        executor=g_quant_config.executor, \
        scope=g_quant_config.scope, \
        model_dir=g_quant_config.float_infer_model_path, \
        quantize_model_path=g_quant_config.quantize_model_path, \
        sample_generator=g_quant_config.train_sample_generator, \
        model_filename=g_quant_config.model_filename, \
        params_filename=g_quant_config.params_filename, \
        save_model_filename=g_quant_config.save_model_filename, \
        save_params_filename=g_quant_config.save_params_filename, \
        quantizable_op_type=g_quant_config.quantizable_op_type, \
        activation_quantize_type=activation_quantize_method, \
        weight_quantize_type=g_quant_config.weight_quantize_type, \
        algo=algo, \
        hist_percent=hist_percent, \
        bias_correction=bias_correct, \
        batch_size=batch_size, \
        batch_nums=batch_num)
    
    global g_min_emd_loss
    emd_loss = eval_quant_model()
    if emd_loss < g_min_emd_loss:
        g_min_emd_loss = emd_loss
        if os.path.exists(g_quant_config.quantize_model_path):
            shutil.rmtree(g_quant_config.quantize_model_path)
        os.system(quant_model_cache, g_quant_config.quantize_model_path)
    return mse_loss
    
def quant_post_hpo(
    executor,
    place,
    model_dir,
    quantize_model_path,
    train_sample_generator=None,
    eval_sample_generator=None,
    model_filename=None,
    params_filename=None,
    save_model_filename='__model__',
    save_params_filename='__params__',
    scope=None,
    quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
    is_full_quantize=False,
    weight_bits=8,
    activation_bits=8,
    weight_quantize_type='channel_wise_abs_max',
    optimize_model=False,
    is_use_cache_file=False,
    cache_dir="./temp_post_training",
    runcount_limit=30):
    """doc"""
    global g_quant_config
    g_quant_config = QuantConfig(
        executor,
        place,
        model_dir,
        quantize_model_path,
        train_sample_generator,
        eval_sample_generator,
        model_filename,
        params_filename,
        save_model_filename,
        save_params_filename,
        scope,
        quantizable_op_type,
        is_full_quantize,
        weight_bits,
        activation_bits,
        weight_quantize_type,
        optimize_model,
        is_use_cache_file,
        cache_dir)
    cs = ConfigurationSpace()

    algo = CategoricalHyperparameter("algo", ["KL", "hist", "avg", "mse"], default_value="KL")
    bias_correct = CategoricalHyperparameter("bias_correct", [True, False], default_value=False)
    weight_quantize_method = CategoricalHyperparameter("weight_quantize_method", \
        [weight_quantize_type], default_value=weight_quantize_type)
    activation_quantize_method = CategoricalHyperparameter("activation_quantize_method", \
        ['moving_average_abs_max', 'range_abs_max'], \
        default_value="moving_average_abs_max")
    hist_percent = UniformFloatHyperparameter("hist_percent", 0.98, 0.999, default_value=0.99)
    batch_size = UniformIntegerHyperparameter("batch_size", 10, 30, default_value=10)
    batch_num = UniformIntegerHyperparameter("batch_num", 10, 30, default_value=10)
        
    cs.add_hyperparameters([algo, bias_correct, weight_quantize_method, activation_quantize_method, \
                            hist_percent, batch_size, batch_num])

    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative runtime)
                     "runcount-limit": runcount_limit,  # max. number of function evaluations; for this example set to a low number
                     "cs": cs,  # configuration space
                     "deterministic": "True",
                     "limit_resources": "False",
                     "memory_limit": 4096  # adapt this to reasonable value for your hardware
                     })

    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
              tae_runner=quantize)

    # Example call of the function with default values
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = smac.get_tae_runner().run(cs.get_default_configuration(), 1)[1]
    print("Value for default configuration: %.2f" % def_value)

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = smac.get_tae_runner().run(incumbent, 1)[1]
    print("Optimized Value: %.2f" % inc_value)
    print("quantize completed")

