# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

import logging
import paddle
from ..quant.quanter import quant_aware, _quant_config_default, _parse_configs, pact, get_pact_optimizer
from ..dist import *
from ..common.recover_program import _remove_fetch_node
from ..common import get_logger
from .strategy_config import ProgramInfo

_logger = get_logger(__name__, level=logging.INFO)
__all__ = [
    'build_distill_program', 'build_quant_program', 'build_prune_program'
]


def _create_optimizer(train_config):
    """create optimizer"""
    ### TODO: support more optimizer
    optimizer = paddle.optimizer.SGD(
        learning_rate=train_config["learning_rate"],
        weight_decay=paddle.regularizer.L2Decay(train_config["weight_decay"]))
    return optimizer


def _parse_distill_loss(distill_node_pair):
    """parse distill loss config"""
    assert len(distill_node_pair) % 2 == 0, \
        "distill_node_pair config wrong, the length needs to be an even number"
    _logger.info("train config.distill_node_pair: {}".format(distill_node_pair))
    distill_loss = 0
    for i in range(len(distill_node_pair) // 2):
        ### TODO: support more loss
        distill_loss += l2_loss(distill_node_pair[i * 2],
                                distill_node_pair[i * 2 + 1])
    return distill_loss


def _load_program_and_merge(train_program, config, teacher_idx=None):
    [teacher_program, teacher_feed_target_names, teacher_fetch_targets]= paddle.static.load_inference_model( \
        path_prefix=teacher_model_dir, \
        model_filename=config["teacher_model_filename"] if "teacher_model_filename" in config else None, \
        params_filename=config["teacher_params_filename"] if "teacher_params_filename" in config else None, \
        executor=executor)
    _remove_fetch_node(teacher_program)

    test_program = train_program.clone(for_test=True)

    data_name_map = {}
    assert len(feed_target_names) == len(teacher_feed_target_names), \
        "the number of feed nodes in the teacher model is not equal to the student model"
    for i, name in enumerate(feed_target_names):
        data_name_map[teacher_feed_target_names[i]] = name

    if teacher_idx is None:
        teacher_name_prefix = 'teacher_'
    else:
        teacher_name_prefix = 'teacher{}_'.format(str(teacher_idx))

    merge(
        teacher_program,
        train_program,
        data_name_map,
        place,
        name_prefix=teacher_name_prefix,
        merge_feed=config['merge_feed'])
    return train_program


def build_distill_program(executor,
                          place,
                          config,
                          train_config,
                          train_program_info=None,
                          pruner=None):
    """build distill program with infermodel"""
    if train_program_info is None:
        [train_program, feed_target_names, fetch_targets]= paddle.static.load_inference_model( \
            path_prefix=config["model_dir"] if "model_dir" in config else config["model_path_prefix"], \
            model_filename=config["model_filename"] if "model_filename" in config else None, \
            params_filename=config["params_filename"] if "params_filename" in config else None, \
            executor=executor)
        train_program = recover_inference_program(train_program)
    else:
        startup_program = paddle.static.Program()
        train_program = train_program_info.program
        feed_target_names = train_program_info.feed_target_names
        fetch_targets = train_program_info.fetch_targets

    teacher_model_dir = config[
        "teacher_model_dir"] if "teacher_model_dir" in config else config[
            "teacher_model_path_prefix"]
    if isinstance(teacher_model_dir, list):
        for tea_idx in range(len(teacher_model_dir)):
            train_program = _load_program_and_merge(
                train_program, config, teacher_idx=(tea_idx + 1))
    else:
        train_program = _load_program_and_merge(
            train_program, config, teacher_idx=None)
    # all feed node should set stop_gradient is False, for using pact quant algo.
    for var in train_program.list_vars():
        if var.name in data_name_map.values() or var.name in data_name_map.keys(
        ):
            var.stop_gradient = False

    train_fetch_list = []
    with paddle.static.program_guard(train_program, startup_program):
        with paddle.utils.unique_name.guard('merge'):
            optimizer = _create_optimizer(train_config)

            distill_loss = _parse_distill_loss(config['distill_node_pair'])
            loss = paddle.mean(distill_loss)
            loss.stop_gradient = False

            if 'prune_algo' in config:  ### prune & asp
                if config['prune_algo'] == 'asp':
                    optimizer = pruner.decorate(optimizer)
                optimizer.minimize(loss)
            elif 'prune_strategy' in config:  ###unstructure prune
                optimizer.minimize(loss, no_grad_set=pruner.no_grad_set)
            else:
                optimizer.minimize(loss)

            train_fetch_list.append(loss)

    train_program_info = ProgramInfo(startup_program, train_program,
                                     feed_target_names, train_fetch_list,
                                     optimizer)
    test_program_info = ProgramInfo(startup_program, test_program,
                                    feed_target_names, fetch_targets)
    return train_program_info, test_program_info


def build_quant_program(executor, place, config, train_program_info,
                        test_program_info):
    scope = paddle.static.global_scope()

    assert isinstance(config, dict), "quant config must be dict"
    default_config = _quant_config_default
    default_config.update(config)
    config = _parse_configs(default_config)

    use_pact = config["use_pact"]
    if use_pact:
        act_preprocess_func = pact
        optimizer_func = get_pact_optimizer
        pact_executor = executor
    else:
        act_preprocess_func = None
        optimizer_func = None
        pact_executor = None

    test_program = quant_aware(
        test_program_info.program,
        place,
        config,
        scope=scope,
        act_preprocess_func=None,
        optimizer_func=None,
        executor=None,
        for_test=True)

    train_program = quant_aware(
        train_program_info.program,
        place,
        config,
        scope=scope,
        act_preprocess_func=act_preprocess_func,
        optimizer_func=optimizer_func,
        executor=pact_executor,
        for_test=False,
        return_program=True)

    train_program_info = train_program_info._replace(program=train_program)
    test_program_info = test_program_info._replace(program=test_program)
    return train_program_info, test_program_info, config


def build_prune_program(executor, place, config, train_program_info, strategy):
    if 'unstructure' in strategy:
        from ..prune.unstructured_pruner import UnstructuredPruner, GMPUnstructuredPruner
        if config["prune_strategy"] is None:
            pruner = UnstructuredPruner(
                train_program_info.program,
                mode=config['prune_mode'],
                ratio=config['prune_ratio'],
                threshold=config['threshold'],
                prune_params_type=config['prune_params_type'],
                place=place,
                local_sparsity=config['local_sparsity'], )
        elif config["prune_strategy"] == "gmp":
            pruner = GMPUnstructuredPruner(
                train_program_info.program,
                ratio=config['prune_ratio'],
                threshold=config['threshold'],
                prune_params_type=config['prune_params_type'],
                place=place,
                local_sparsity=config['local_sparsity'],
                config=config['gmp_config'])
    else:
        if config['prune_algo'] == 'prune':
            from ..prune import Pruner
            pruner = Pruner(config["criterion"])
            params = []
            ### TODO: fix
            for param in train_program_info.program.global_block(
            ).all_parameters():
                if param.name in config['prune_params_name']:
                    params.append(param.name)

            pruned_program, _, _ = pruner.prune(
                val_program,
                paddle.static.global_scope(),
                params=params,
                ratios=config['pruned_ratio'] * len(params),
                place=place)
            train_program_info = train_program_info._replace(
                program=pruned_program)

        elif config['prune_algo'] == 'asp':
            from paddle.static import sparsity
            excluded_params_name = []
            for param in train_program_info.program.global_block(
            ).all_parameters():
                if param.name not in config['prune_params_name']:
                    excluded_params_name.append(param.name)
            sparsity.set_excluded_layers(train_program_info.program,
                                         excluded_params_name)
            pruner = sparsity
        else:
            raise NotImplementedError(
                "prune_algo must be choice in [\"prune\", \"asp\"], {} is not support".
                format(config['prune_algo']))

    return pruner, train_program_info
