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
"""train aware quant with infermodel"""

import copy
import os
import argparse
import json
import six
from collections import namedtuple
import time
import shutil
import numpy as np
import paddle
from ..common.recover_program import recover_inference_program
from .quanter import _quant_config_default, _parse_configs, pact, get_pact_optimizer
from .quanter import quant_aware, convert
from ..dist import merge, l2, soft_label
from ..auto_compression.create_compressed_program import build_distill_program
import logging
from ..common import get_logger
_logger = get_logger(__name__, level=logging.INFO)

############################################################################################################
# quantization training configs
############################################################################################################
_train_config_default = {
    # configs of training aware quantization with infermodel
    "num_epoch":
    1000,  # training epoch num
    "max_iter":
    -1,  # max training iteration num
    "save_iter_step":
    1000,  # save quant model checkpoint every save_iter_step iteration
    "learning_rate":
    0.0001,  # learning rate
    "weight_decay":
    0.0001,  # weight decay
    "use_pact":
    False,  # use pact quantization or not
    # quant model checkpoints save path
    "quant_model_ckpt_path":
    "./quant_model_checkpoints/",
    # storage directory of teacher model + teacher model name (excluding suffix)
    "teacher_model_path_prefix":
    None,
    # storage directory of model + model name (excluding suffix)
    "model_path_prefix":
    None,
    """ distillation node configuration: 
        the name of the distillation supervision nodes is configured as a list, 
        and the teacher node and student node are arranged in pairs.
        for example, ["teacher_fc_0.tmp_0", "fc_0.tmp_0", "teacher_batch_norm_24.tmp_4", "batch_norm_24.tmp_4"]
    """
    "node":
    None
}


def _parse_train_configs(train_config):
    """
    check if user's train configs are valid.
    Args:
        train_config(dict): user's train config.
    Return:
        configs(dict): final configs will be used.
    """

    configs = copy.deepcopy(_train_config_default)
    configs.update(train_config)

    assert isinstance(configs['num_epoch'], int), \
        "'num_epoch' must be int value"
    assert isinstance(configs['max_iter'], int), \
        "'max_iter' must be int value"
    assert isinstance(configs['save_iter_step'], int), \
        "'save_iter_step' must be int value"
    assert isinstance(configs['learning_rate'], float), \
        "'learning_rate' must be float"
    assert isinstance(configs['weight_decay'], float), \
        "'weight_decay' must be float"
    assert isinstance(configs['use_pact'], bool), \
        "'use_pact' must be bool"
    assert isinstance(configs['quant_model_ckpt_path'], str), \
        "'quant_model_ckpt_path' must be str"
    assert isinstance(configs['teacher_model_path_prefix'], str), \
        "'teacher_model_path_prefix' must both be string"
    assert isinstance(configs['model_path_prefix'], str), \
        "'model_path_prefix' must both be str"
    assert isinstance(configs['node'], list), \
        "'node' must both be list"
    assert len(configs['node']) > 0, \
        "'node' not configured with distillation nodes"
    return train_config


def _compile_program(program, fetch_var_name):
    """compiling program"""

    build_strategy = paddle.static.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = False
    build_strategy.fuse_all_reduce_ops = False
    build_strategy.sync_batch_norm = False
    compiled_prog = paddle.static.CompiledProgram(
        program, build_strategy=build_strategy)
    return compiled_prog


def quant_aware_with_infermodel(executor,
                                place,
                                scope=None,
                                train_reader=None,
                                quant_config=None,
                                train_config=None,
                                test_callback=None):
    """train aware quantization with infermodel
    Args:
        executor(paddle.static.Executor): The executor to load, run and save the
            quantized model.
        place(paddle.CPUPlace or paddle.CUDAPlace): This parameter represents
            the executor run on which device.
        scope(paddle.static.Scope, optional):  Scope records the mapping between
            variable names and variables, similar to brackets in
            programming languages. Usually users can use
            `paddle.static.global_scope <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_.
            When ``None`` will use
            `paddle.static.global_scope() <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_
            . Default: ``None``.
        train_reader(data generator): data generator, yield feed_dictionary, {feed_name[0]:data[0], feed_name[1]:data[1]}.
        quant_config(dict, optional): configs for convert. if set None, will use
                default config. It must be same with config that used in
                'quant_aware'. Default is None.
        train_config(dict):train aware configs, include num_epoch, save_iter_step, learning_rate,
                weight_decay, use_pact, quant_model_ckpt_path,
                model_path_prefix, teacher_model_path_prefix,
                node(node_name1, node_name2, ...)
        test_callback(callback function): callback function include two params: compiled test quant program and checkpoint save filename.
                user can implement test logic.
    Returns:
        None
    """
    scope = paddle.static.global_scope() if not scope else scope
    # parse quant config
    if quant_config is None:
        quant_config = _quant_config_default
    else:
        assert isinstance(quant_config, dict), "quant config must be dict"
        quant_config = _parse_configs(quant_config)
    _logger.info("quant_aware config {}".format(quant_config))

    train_config = _parse_train_configs(train_config)
    distill_program_info, test_program_info = build_distill_program(
        executor, place, train_config, train_config)
    startup_program = distill_program_info.startup_program
    train_program = distill_program_info.program
    train_feed_names = distill_program_info.feed_target_names
    train_fetch_list = distill_program_info.fetch_targets
    optimizer = distill_program_info.optimizer
    test_program = test_program_info.program
    test_feed_names = test_program_info.feed_target_names
    test_fetch_list = test_program_info.fetch_targets

    ############################################################################
    # quant
    ############################################################################
    use_pact = train_config["use_pact"]
    if use_pact:
        act_preprocess_func = pact
        optimizer_func = get_pact_optimizer
        pact_executor = executor
    else:
        act_preprocess_func = None
        optimizer_func = None
        pact_executor = None

    test_program = quant_aware(
        test_program,
        place,
        quant_config,
        scope=scope,
        act_preprocess_func=None,
        optimizer_func=None,
        executor=None,
        for_test=True)
    train_program = quant_aware(
        train_program,
        place,
        quant_config,
        scope=scope,
        act_preprocess_func=act_preprocess_func,
        optimizer_func=optimizer_func,
        executor=pact_executor,
        for_test=False,
        return_program=True)

    executor.run(startup_program)
    compiled_train_prog = _compile_program(train_program,
                                           train_fetch_list[0].name)
    compiled_test_prog = _compile_program(test_program, test_fetch_list[0].name)
    num_epoch = train_config["num_epoch"]
    save_iter_step = train_config["save_iter_step"]
    iter_sum = 0
    for epoch in range(num_epoch):
        for iter_num, feed_dict in enumerate(train_reader()):
            np_probs_float = executor.run(compiled_train_prog, \
                feed=feed_dict, \
                fetch_list=train_fetch_list)
            print("loss: ", np_probs_float)

            if iter_num > 0 and iter_num % save_iter_step == 0:
                checkpoint_name = "epoch_" + str(epoch) + "_iter_" + str(
                    iter_num)
                if not os.path.exists(train_config["quant_model_ckpt_path"]):
                    os.makedirs(train_config["quant_model_ckpt_path"])
                paddle.static.save(
                    program=test_program,
                    model_path=os.path.join(
                        train_config["quant_model_ckpt_path"], checkpoint_name))
                test_callback(compiled_test_prog, test_feed_names,
                              test_fetch_list, checkpoint_name)
            iter_sum += 1
            if train_config["max_iter"] >= 0 and iter_sum > train_config["max_iter"]:
                return


def export_quant_infermodel(
        executor,
        place=None,
        scope=None,
        quant_config=None,
        train_config=None,
        checkpoint_path=None,
        export_inference_model_path_prefix="./export_quant_infermodel"):
    """export quant model checkpoints to infermodel.
    Args:
        executor(paddle.static.Executor): The executor to load, run and save the
            quantized model.
        place(paddle.CPUPlace or paddle.CUDAPlace): This parameter represents
            the executor run on which device.
        scope(paddle.static.Scope, optional):  Scope records the mapping between
            variable names and variables, similar to brackets in
            programming languages. Usually users can use
            `paddle.static.global_scope <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_.
            When ``None`` will use
            `paddle.static.global_scope() <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_
            . Default: ``None``.
        quant_config(dict, optional): configs for convert. if set None, will use
                default config. It must be same with config that used in
                'quant_aware'. Default is None.
        train_config(dict):train aware configs, include num_epoch, save_iter_step, learning_rate,
                weight_decay, use_pact, quant_model_ckpt_path,
                model_path_prefix, teacher_model_path_prefix, 
                node(node_name1, node_name2, ...)
        checkpoint_path(str): checkpoint path need to export quant infer model.
        export_inference_model_path_prefix(str): export infer model path prefix, storage directory of model + model name (excluding suffix).
    Returns:
        None
    """
    scope = paddle.static.global_scope() if not scope else scope
    # parse quant config
    if quant_config is None:
        quant_config = _quant_config_default
    else:
        assert isinstance(quant_config, dict), "quant config must be dict"
        quant_config = _parse_configs(quant_config)
    _logger.info("quant_aware config {}".format(quant_config))

    train_config = _parse_train_configs(train_config)
    _, test_program_info = build_distill_program(executor, place, train_config,
                                                 train_config)
    test_program = test_program_info.program
    test_feed_names = test_program_info.feed_target_names
    test_fetch_list = test_program_info.fetch_targets

    ############################################################################
    # quant
    ############################################################################
    use_pact = False  # export model should set use_pact is False
    if use_pact:
        act_preprocess_func = pact
        optimizer_func = get_pact_optimizer
        pact_executor = executor
    else:
        act_preprocess_func = None
        optimizer_func = None
        pact_executor = None

    test_program = quant_aware(
        test_program,
        place,
        quant_config,
        scope=scope,
        act_preprocess_func=act_preprocess_func,
        optimizer_func=optimizer_func,
        executor=pact_executor,
        for_test=True)

    paddle.static.load(
        executor=executor,
        model_path=os.path.join(checkpoint_path),
        program=test_program)
    ############################################################################################################
    # 3. Freeze the graph after training by adjusting the quantize
    #    operators' order for the inference.
    #    The dtype of float_program's weights is float32, but in int8 range.
    ############################################################################################################
    float_program = convert(test_program, place, quant_config, scope=scope)
    ############################################################################################################
    # 4. Save inference model
    ############################################################################################################
    export_model_dir = os.path.abspath(
        os.path.join(export_inference_model_path_prefix, os.path.pardir))
    if not os.path.exists(export_model_dir):
        os.makedirs(export_model_dir)

    feed_vars = []
    for name in test_feed_names:
        for var in float_program.list_vars():
            if var.name == name:
                feed_vars.append(var)
                break
    assert len(feed_vars) > 0, "can not find feed vars in quant program"
    paddle.static.save_inference_model(
        path_prefix=export_inference_model_path_prefix,
        feed_vars=feed_vars,
        fetch_vars=test_fetch_list,
        executor=executor,
        program=float_program)
