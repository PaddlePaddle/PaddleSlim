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
from collections import namedtuple
import time
import shutil
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import Parameter
from paddleslim.dist import merge, l2_loss, soft_label_loss, fsp_loss
from paddleslim.core import GraphWrapper
from paddleslim.quant import quant_aware, convert
from .quanter import _quant_config_default, _parse_configs
import logging
logging.getLogger().setLevel(logging.INFO)
from ..common import get_logger
_logger = get_logger(__name__, level=logging.INFO)

############################################################################################################
# quantization training configs
############################################################################################################
_train_config_default = {
    # configs of training aware quantization with infermodel
    "num_epoch": 1000,  # training epoch num
    "max_iter": -1,
    "save_iter_step": 1000,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "use_pact": False,
    "quant_model_ckpt_path": "./quant_model_checkpoints/",
    "teacher_model_path": None,
    "teacher_model_filename": "__model__",
    "teacher_params_filename": None,
    "model_path": None,
    "model_filename": "__model__",
    "params_filename": None,
    "distill_node_pair": None
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
        "'num_epoch' must be int value'"
    assert isinstance(configs['max_iter'], int), \
        "'max_iter' must be int value'"
    assert isinstance(configs['save_iter_step'], int), \
        "'save_iter_step' must be int value'"
    assert isinstance(configs['learning_rate'], float), \
        "'learning_rate' must be float'"
    assert isinstance(configs['weight_decay'], float), \
        "'weight_decay' must be float'"
    assert isinstance(configs['use_pact'], bool), \
        "'use_pact' must be bool'"
    assert isinstance(configs['quant_model_ckpt_path'], str), \
        "'quant_model_ckpt_path' must be str'"
    assert isinstance(configs['teacher_model_path'], str), \
        "'teacher_model_path' must both be float'"
    assert isinstance(configs['model_path'], str), \
        "'model_path' must both be str'"
    return train_config


def _create_optimizer(train_config):
    """create optimizer"""
    optimizer = paddle.optimizer.SGD(
        learning_rate=train_config["learning_rate"],
        weight_decay=paddle.regularizer.L2Decay(train_config["weight_decay"]))
    return optimizer


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


def _parse_distill_loss(train_config):
    """parse distill loss config"""
    assert len(train_config["distill_node_pair"]) % 2 == 0, \
        "distill_node_pair config wrong, the length needs to be an even number"
    print("train config.distill_node_pair: ", train_config["distill_node_pair"])
    distill_loss = 0
    for i in range(len(train_config["distill_node_pair"]) // 2):
        print(train_config["distill_node_pair"][i * 2],
              train_config["distill_node_pair"][i * 2 + 1])
        distill_loss += l2_loss(train_config["distill_node_pair"][i * 2],
                                train_config["distill_node_pair"][i * 2 + 1])
    print(distill_loss)
    return distill_loss

DistillProgramInfo = namedtuple("DistillProgramInfo", \
    "startup_program train_program train_feed_names train_fetch_list \
     optimizer test_program test_feed_names test_fetch_list"
                                                            )


def build_distill_prog_with_infermodel(executor, place, train_config):
    """build distill program with infermodel"""
    [train_program, feed_target_names, fetch_targets]= fluid.io.load_inference_model( \
        dirname=train_config["model_path"], \
        executor=executor, \
        model_filename=train_config["model_filename"], \
        params_filename=train_config["params_filename"])
    _remove_fetch_node(train_program)
    [teacher_program, teacher_feed_target_names, teacher_fetch_targets]= fluid.io.load_inference_model( \
        dirname=train_config["teacher_model_path"], \
        executor=executor, \
        model_filename=train_config["teacher_model_filename"], \
        params_filename=train_config["teacher_params_filename"])
    _remove_fetch_node(teacher_program)
    test_program = train_program.clone(for_test=True)

    train_program = _recover_param_attr(train_program)
    for var in train_program.list_vars():
        var.stop_gradient = False
    train_graph = GraphWrapper(train_program)
    for op in train_graph.ops():
        op._op._set_attr("is_test", False)

    ############################################################################
    # distill
    ############################################################################
    data_name_map = {}
    assert len(feed_target_names) == len(teacher_feed_target_names), \
        "the number of feed nodes in the teacher model is not equal to the student model"
    for i, name in enumerate(feed_target_names):
        data_name_map[teacher_feed_target_names[i]] = name
    merge(teacher_program, train_program, data_name_map, place)

    # all feed node should set stop_gradient is False, for using pact quant algo.
    for var in train_program.list_vars():
        if var.name in data_name_map.values() or var.name in data_name_map.keys(
        ):
            var.stop_gradient = False

    train_fetch_list = []
    train_fetch_name_list = []
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(train_program, startup_program):
        with fluid.unique_name.guard('merge'):
            optimizer = _create_optimizer(train_config)

            distill_loss = _parse_distill_loss(train_config)
            loss = paddle.mean(distill_loss)
            loss.stop_gradient = False
            p_g_list = paddle.static.append_backward(loss=loss)
            opts = optimizer.apply_gradients(p_g_list)

            train_fetch_list.append(loss)
            train_fetch_name_list.append(loss.name)

    return DistillProgramInfo(startup_program, train_program, \
        feed_target_names, train_fetch_list, optimizer, \
        test_program, feed_target_names, fetch_targets)


def _compile_program(program, fetch_var_name):
    """compiling program"""
    compiled_prog = paddle.static.CompiledProgram(program)
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = False
    build_strategy.fuse_all_reduce_ops = False
    build_strategy.sync_batch_norm = False
    exec_strategy = paddle.static.ExecutionStrategy()
    compiled_prog = compiled_prog.with_data_parallel(
        loss_name=fetch_var_name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)
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
                weight_decay, use_pact, quant_model_ckpt_path, teacher_model_path, teacher_model_filename,
                teacher_params_filename, model_path, model_filename, params_filename,
                distill_node_pair(teacher_node_name1, node_name1, teacher_node_name2, teacher_node_name2, ...)
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
    distill_program_info = build_distill_prog_with_infermodel(executor, place,
                                                              train_config)
    startup_program = distill_program_info.startup_program
    train_program = distill_program_info.train_program
    train_feed_names = distill_program_info.train_feed_names
    train_fetch_list = distill_program_info.train_fetch_list
    optimizer = distill_program_info.optimizer
    test_program = distill_program_info.test_program
    test_feed_names = distill_program_info.test_feed_names
    test_fetch_list = distill_program_info.test_fetch_list

    ############################################################################
    # quant
    ############################################################################
    def pact(x):
        """clip feature value range"""
        helper = LayerHelper("pact", **locals())
        dtype = 'float32'
        init_thres = 16
        u_param_attr = paddle.ParamAttr(
            name=x.name + '_pact',
            initializer=paddle.nn.initializer.Constant(value=init_thres),
            regularizer=paddle.regularizer.L2Decay(0.0001),
            learning_rate=1)
        u_param = helper.create_parameter(
            attr=u_param_attr, shape=[1], dtype=dtype)

        part_a = paddle.nn.functional.relu(x - u_param)
        part_b = paddle.nn.functional.relu(-u_param - x)
        x = x - part_a + part_b
        return x

    def get_optimizer():
        """optimizer for pact params"""
        return paddle.optimizer.Momentum(0.0001, 0.9)

    use_pact = train_config["use_pact"]
    if use_pact:
        act_preprocess_func = pact
        optimizer_func = get_optimizer
        pact_executor = executor
    else:
        act_preprocess_func = None
        optimizer_func = None
        pact_executor = None

    test_program = quant_aware(
        test_program,
        place,
        quant_config,
        scope=None,
        act_preprocess_func=act_preprocess_func,
        optimizer_func=optimizer_func,
        executor=pact_executor,
        for_test=True)
    train_program = quant_aware(
        train_program,
        place,
        quant_config,
        scope=None,
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
                test_callback(compiled_test_prog, test_feed_names,
                              test_fetch_list, checkpoint_name)
                paddle.static.save(
                    program=test_program,
                    model_path=os.path.join(
                        train_config["quant_model_ckpt_path"], checkpoint_name))
            iter_sum += 1
            if train_config["max_iter"] >= 0 and iter_sum > train_config[
                    "max_iter"]:
                return


def export_quant_infermodel(
        executor,
        place=None,
        scope=None,
        quant_config=None,
        train_config=None,
        checkpoint_path=None,
        export_infermodel_path="./export_quant_infermodel/"):
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
                weight_decay, use_pact, quant_model_ckpt_path, teacher_model_path, teacher_model_filename,
                teacher_params_filename, model_path, model_filename, params_filename,
                distill_node_pair(teacher_node_name1, node_name1, teacher_node_name2, teacher_node_name2, ...)
        checkpoint_path(str): checkpoint path need to export quant infer model.
        export_infermodel_path(str): export infer model path.
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
    distill_program_info = build_distill_prog_with_infermodel(executor, place,
                                                              train_config)
    test_program = distill_program_info.test_program
    test_feed_names = distill_program_info.test_feed_names
    test_fetch_list = distill_program_info.test_fetch_list

    ############################################################################
    # quant
    ############################################################################
    def pact(x):
        """clip feature value range"""
        helper = LayerHelper("pact", **locals())
        dtype = 'float32'
        init_thres = 16
        u_param_attr = paddle.ParamAttr(
            name=x.name + '_pact',
            initializer=paddle.nn.initializer.Constant(value=init_thres),
            regularizer=paddle.regularizer.L2Decay(0.0001),
            learning_rate=1)
        u_param = helper.create_parameter(
            attr=u_param_attr, shape=[1], dtype=dtype)

        part_a = paddle.nn.functional.relu(x - u_param)
        part_b = paddle.nn.functional.relu(-u_param - x)
        x = x - part_a + part_b
        return x

    def get_optimizer():
        """optimizer for pact params"""
        return paddle.optimizer.Momentum(0.0001, 0.9)

    use_pact = train_config["use_pact"]
    if use_pact:
        act_preprocess_func = pact
        optimizer_func = get_optimizer
        pact_executor = executor
    else:
        act_preprocess_func = None
        optimizer_func = None
        pact_executor = None

    test_program = quant_aware(
        test_program,
        place,
        quant_config,
        scope=None,
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
    float_program, int8_program = convert(test_program, place, quant_config, \
                                                        scope=scope, \
                                                        save_int8=True)
    ############################################################################################################
    # 4. Save inference model
    ############################################################################################################
    model_path = export_infermodel_path
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    paddle.fluid.io.save_inference_model(
        dirname=model_path,
        feeded_var_names=test_feed_names,
        target_vars=test_fetch_list,
        executor=executor,
        main_program=float_program,
        model_filename=model_path + '/model',
        params_filename=model_path + '/params')
