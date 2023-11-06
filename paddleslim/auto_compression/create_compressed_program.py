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
import numpy as np
import paddle
import paddle.distributed.fleet as fleet
import paddle.optimizer as optimizer
import paddle.regularizer as regularizer
from ..quant.quanter import quant_aware, _quant_config_default, _parse_configs, pact, get_pact_optimizer
from ..dist import *
from ..common.recover_program import recover_inference_program, _remove_fetch_node
from ..common import get_logger
from .strategy_config import ProgramInfo
from ..common.load_model import load_inference_model
from ..analysis import flops

_logger = get_logger(__name__, level=logging.INFO)
__all__ = [
    'build_distill_program', 'build_quant_program', 'build_prune_program',
    'remove_unused_var_nodes'
]


def _create_lr_scheduler(train_config):
    if 'learning_rate' not in train_config:
        raise RuntimeError(
            'No `learning_rate` specified in the configuration file.')
    if isinstance(train_config.get('learning_rate'), float):
        return train_config.get('learning_rate')

    params = train_config.get('learning_rate')
    lr_type = params.pop('type')
    return getattr(optimizer.lr, lr_type)(**params)


def _create_optimizer(train_config):
    """create optimizer"""
    if 'optimizer_builder' not in train_config:
        train_config['optimizer_builder'] = {'optimizer': {'type': 'SGD'}}

    optimizer_builder = train_config['optimizer_builder']
    assert isinstance(
        optimizer_builder, dict
    ), "Value of 'optimizer_builder' in train_config should be dict but got {}".format(
        type(optimizer_builder))
    if 'grad_clip' in optimizer_builder:
        g_clip_params = optimizer_builder['grad_clip']
        g_clip_type = g_clip_params.pop('type')
        grad_clip = getattr(paddle.nn, g_clip_type)(**g_clip_params)
    else:
        grad_clip = None

    ### build regularization
    if 'regularizer' in optimizer_builder:
        reg_params = optimizer_builder['regularizer']
        reg_type = reg_params.pop('type')
        reg = getattr(regularizer, reg_type)(**reg_params)
    elif 'weight_decay' in optimizer_builder:
        reg = optimizer_builder.pop('weight_decay')
    else:
        reg = None

    ### build learning rate
    lr = _create_lr_scheduler(train_config)

    ### build optimizer
    optim_params = optimizer_builder['optimizer']
    optim_type = optim_params.pop('type')
    opt = getattr(optimizer, optim_type)(
        learning_rate=lr, grad_clip=grad_clip, weight_decay=reg, **optim_params)
    return opt, lr


def _find_var_from_program(program, var_name):
    for block in program.blocks:
        if block.has_var(var_name):
            return block.var(var_name)
    raise ValueError("var {} not in this program".format(var_name))


def _get_distill_node(student_program, config):
    node = config.get('node')
    if len(node) == 0:
        return None

    ### the type of node is list or list(list)
    if isinstance(node[0], list):
        test_node = node[0][0]
    else:
        test_node = node[0]
    try:
        test_var = _find_var_from_program(student_program, test_node)
        distill_node_pair = []
        if isinstance(node[0], list):
            for n_list in node:
                tmp_node_pair = []
                for n in n_list:
                    tmp_node_pair.append('teacher_' + n)
                    tmp_node_pair.append(n)
                distill_node_pair.append(tmp_node_pair)
        else:
            for n in node:
                distill_node_pair.append('teacher_' + n)
                distill_node_pair.append(n)
        return distill_node_pair
    except:
        return node


def _get_target_node(distill_node, teacher=False):
    tmp_nodes = set()
    if isinstance(distill_node[0], list):
        for n_list in distill_node:
            for n in n_list:
                tmp_nodes.add(n)
    else:
        for n in distill_node:
            tmp_nodes.add(n)

    targets = []
    for node in tmp_nodes:
        if teacher and 'teacher_' in node:
            tmp = node.split('teacher_')[-1]
            targets.append(tmp)
        if not teacher and 'teacher_' not in node:
            targets.append(node)

    return targets


def _parse_distill_loss(distill_node_pair,
                        distill_loss='l2',
                        distill_lambda=1.0):
    """parse distill loss config"""
    loss_dist = 0.0
    losses = {}
    if isinstance(distill_node_pair[0], str):
        assert isinstance(distill_loss, str)
        assert isinstance(distill_lambda, float)
        distill_node_pair = [distill_node_pair]
        distill_loss = [distill_loss]
        distill_lambda = [distill_lambda]

    assert len(distill_node_pair) == len(distill_loss)
    assert len(distill_node_pair) == len(distill_lambda)
    for node, loss_clas, lam in zip(distill_node_pair, distill_loss,
                                    distill_lambda):
        tmp_loss = losses.get(loss_clas, 0.0)
        _logger.info(
            "train config.distill_node_pair: {}".format(node, loss_clas, lam))
        assert len(node) % 2 == 0, \
            "distill_node_pair config wrong, the length needs to be an even number"
        for i in range(len(node) // 2):
            tmp_loss += eval(loss_clas)(node[i * 2], node[i * 2 + 1]) * lam
        loss_dist += tmp_loss
        losses[loss_clas] = tmp_loss

    return loss_dist, losses


def _load_program_and_merge(executor,
                            place,
                            train_program,
                            config,
                            model_dir,
                            model_filename,
                            params_filename,
                            distill_node_pair,
                            teacher_idx=None,
                            feed_target_names=None):
    scope = paddle.static.global_scope()
    new_scope = paddle.static.Scope()

    if params_filename == 'None':
        params_filename = None

    if params_filename is None and model_filename is not None:
        raise NotImplementedError(
            "NOT SUPPORT parameters saved in separate files. Please convert it to single binary file first."
        )

    with paddle.static.scope_guard(new_scope):
        [teacher_program, teacher_feed_target_names, teacher_fetch_targets]= (load_inference_model( \
            model_dir, \
            model_filename=model_filename, \
            params_filename=params_filename, \
            executor=executor))

    _remove_fetch_node(teacher_program)

    target_nodes = _get_target_node(distill_node_pair, True)
    teacher_program = teacher_program._prune(target_nodes)

    data_name_map = {}

    merge_feed = (
        sorted(feed_target_names) == sorted(teacher_feed_target_names))
    if merge_feed == True:
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
        teacher_scope=new_scope,
        name_prefix=teacher_name_prefix,
        merge_feed=merge_feed)
    if teacher_idx == None or teacher_idx == 1:
        return train_program, data_name_map
    else:
        return train_program, data_name_map


def build_distill_program(executor,
                          place,
                          config,
                          train_config,
                          train_program_info=None,
                          pruner=None,
                          dist_strategy=None,
                          default_distill_node_pair=None):
    """build distill program with infermodel"""
    startup_program = paddle.static.Program()
    if train_program_info is None:
        [train_program, feed_target_names, fetch_targets]= (load_inference_model( \
            path_prefix=config["model_dir"] if "model_dir" in config else config["model_path_prefix"], \
            executor=executor))
        train_program = recover_inference_program(train_program)
    else:
        train_program = train_program_info.program
        feed_target_names = train_program_info.feed_target_names
        fetch_targets = train_program_info.fetch_targets

    distill_node_pair = _get_distill_node(train_program,
                                          config) or default_distill_node_pair

    test_program = train_program.clone(for_test=True)

    target_nodes = _get_target_node(distill_node_pair)

    def _prepend_feed(block, feed_idx, feed_target_names):
        for idx in feed_idx[::-1]:
            block._remove_op(idx)

        feed_var = block.create_var(
            name='feed',
            type=paddle.framework.core.VarDesc.VarType.FEED_MINIBATCH,
            persistable=True, )

        for i, name in enumerate(feed_target_names):
            out = block.var(name)
            block._prepend_op(
                type='feed',
                inputs={'X': [feed_var]},
                outputs={'Out': [out]},
                attrs={'col': i})

    judge_feed_pos = False
    if train_program.desc.block(0).op(0).type() != 'feed':
        judge_feed_pos = True
    if judge_feed_pos:
        feed_idx = []
        for op in train_program.global_block().ops:
            if op.type == 'feed':
                feed_idx.append(op.idx)
        _prepend_feed(train_program.global_block(), feed_idx, feed_target_names)
    train_program = train_program._prune(target_nodes)

    teacher_model_dir = config[
        "teacher_model_dir"] if "teacher_model_dir" in config else config[
            "teacher_model_path_prefix"]
    if isinstance(teacher_model_dir, list):
        for tea_idx in range(len(teacher_model_dir)):
            model_filename = config["teacher_model_filename"][
                tea_idx] if "teacher_model_filename" in config else None
            params_filename = config["teacher_params_filename"][
                tea_idx] if "teacher_params_filename" in config else None
            if tea_idx == 0:
                train_program, data_name_map = _load_program_and_merge(
                    executor,
                    place,
                    train_program,
                    config,
                    teacher_model_dir[tea_idx],
                    model_filename,
                    params_filename,
                    distill_node_pair,
                    teacher_idx=(tea_idx + 1),
                    feed_target_names=feed_target_names)
            else:
                train_program, data_name_map = _load_program_and_merge(
                    executor,
                    place,
                    train_program,
                    config,
                    teacher_model_dir[tea_idx],
                    model_filename,
                    params_filename,
                    distill_node_pair,
                    teacher_idx=(tea_idx + 1),
                    feed_target_names=feed_target_names)

    else:
        model_filename = config[
            "teacher_model_filename"] if "teacher_model_filename" in config else None
        params_filename = config[
            "teacher_params_filename"] if "teacher_params_filename" in config else None
        train_program, data_name_map = _load_program_and_merge(
            executor,
            place,
            train_program,
            config,
            teacher_model_dir,
            model_filename,
            params_filename,
            distill_node_pair,
            teacher_idx=None,
            feed_target_names=feed_target_names)
    # all feed node should set stop_gradient is False, for using pact quant algo.
    for var in train_program.list_vars():
        if var.name in data_name_map.values() or var.name in data_name_map.keys(
        ):
            var.stop_gradient = False

    train_fetch_list = []
    with paddle.static.program_guard(train_program, startup_program):
        with paddle.utils.unique_name.guard('merge'):
            optimizer, learning_rate = _create_optimizer(train_config)

            if dist_strategy is not None:
                optimizer = fleet.distributed_optimizer(optimizer,
                                                        dist_strategy)
            else:
                if train_config.get('amp_config') is not None:
                    custom_white_list = train_config['amp_config'].get(
                        'custom_white_list', None)
                    if custom_white_list is not None:
                        train_config['amp_config'].pop('custom_white_list')

                    custom_black_list = train_config['amp_config'].get(
                        'custom_black_list', None)
                    if custom_black_list is not None:
                        train_config['amp_config'].pop('custom_black_list')

                    custom_black_varnames = train_config['amp_config'].get(
                        'custom_black_varnames', None)
                    if custom_black_varnames is not None:
                        train_config['amp_config'].pop('custom_black_varnames')

                    amp_list = paddle.static.amp.CustomOpLists(
                        custom_white_list=custom_white_list,
                        custom_black_list=custom_black_list,
                        custom_black_varnames=custom_black_varnames)
                    optimizer = paddle.static.amp.decorate(
                        optimizer=optimizer,
                        amp_lists=amp_list,
                        init_loss_scaling=128.0,
                        use_dynamic_loss_scaling=True,
                        **train_config['amp_config'])

            distill_loss, loss_dict = _parse_distill_loss(
                distill_node_pair,
                config.get('loss') or 'l2',  ### default loss is l2
                config.get('alpha') or 1.0)  ### default alpha is 1.0
            loss = paddle.mean(distill_loss)
            loss.stop_gradient = False

            if 'prune_params_name' in config:  ### prune
                if 'pruned_ratio' not in config and dist_strategy is None:  ### asp
                    optimizer = pruner.decorate(optimizer)
                optimizer.minimize(loss)
            elif 'prune_strategy' in config:  ###unstructure prune
                optimizer.minimize(loss, no_grad_set=pruner.no_grad_set)
            else:
                optimizer.minimize(loss)

            train_fetch_list.append(loss)

    train_program_info = ProgramInfo(startup_program, train_program,
                                     feed_target_names, train_fetch_list,
                                     optimizer, learning_rate, loss_dict)
    test_program_info = ProgramInfo(startup_program, test_program,
                                    feed_target_names, fetch_targets)
    return train_program_info, test_program_info


def build_quant_program(executor, place, config, train_program_info,
                        test_program_info):
    scope = paddle.static.global_scope()

    assert isinstance(config, dict), "quant config must be dict"

    use_pact = config.pop("use_pact")
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

    train_program_info.program = train_program
    test_program_info.program = test_program
    return train_program_info, test_program_info, config


def _get_label_info(dataloader, feed_target_names):
    label_info = {}
    for data in dataloader():
        if isinstance(data, list) or isinstance(data, tuple):
            data = data[0]
        for key, value in data.items():
            if key in feed_target_names:
                continue
            label_info['name'] = key
            label_info['dtype'] = np.array(value).dtype
            label_info['shape'] = list(np.array(value).shape)
            label_info['shape'][0] = -1
            break
        break
    return label_info


def _get_chn_prune_params(program):
    params = []
    original_shapes = {}
    for block in program.blocks:
        for op in block.ops:
            if op.type == 'conv2d' and op.attr('groups') == 1:
                for inp_name in op.input_arg_names:
                    var_ = block.var(inp_name)
                    if var_.persistable is True:
                        params.append(inp_name)
                        original_shapes[inp_name] = var_.shape
    return params, original_shapes


def _get_asp_prune_params(program):
    params = []
    for block in program.blocks:
        for op in block.ops:
            if (op.type == 'conv2d' and op.attr('groups') == 1
                ) or op.type == 'mul' or op.type == 'matmul_v2':
                for inp_name in op.input_arg_names:
                    var_ = block.var(inp_name)
                    if var_.persistable is True:
                        params.append(inp_name)
    return params


def build_prune_program(executor,
                        place,
                        config,
                        train_program_info,
                        strategy,
                        patterns,
                        eval_dataloader=None):
    if strategy.startswith('unstructure'):
        from ..prune.unstructured_pruner import UnstructuredPruner, GMPUnstructuredPruner
        if config["prune_strategy"] is None:
            pruner = UnstructuredPruner(
                train_program_info.program,
                mode=config['prune_mode'],
                ratio=config['ratio'],
                threshold=config['threshold'],
                prune_params_type=config['prune_params_type'],
                place=place,
                local_sparsity=config['local_sparsity'], )
        elif config["prune_strategy"] == "gmp":
            pruner = GMPUnstructuredPruner(
                train_program_info.program,
                ratio=config['ratio'],
                prune_params_type=config['prune_params_type'],
                place=place,
                local_sparsity=config['local_sparsity'],
                configs=config['gmp_config'])
    elif strategy.startswith('channel_prune'):
        from ..prune import Pruner
        pruner = Pruner(config["criterion"])
        if config['prune_params_name'] is None:
            params, original_shapes = _get_chn_prune_params(
                train_program_info.program)
        else:
            params = []
            original_shapes = {}
            for param in train_program_info.program.global_block(
            ).all_parameters():
                if config['prune_params_name'] is not None and param.name in config['prune_params_name']:
                    params.append(param.name)
                    original_shapes[param.name] = param.shape

        origin_flops = flops(train_program_info.program)

        pruned_program, _, _ = pruner.prune(
            train_program_info.program,
            paddle.static.global_scope(),
            params=params,
            ratios=[config['pruned_ratio']] * len(params) if isinstance(
                config['pruned_ratio'], float) else config['pruned_ratio'],
            place=place)
        _logger.info(
            "####################channel pruning##########################")
        for param in pruned_program.global_block().all_parameters():
            if param.name in original_shapes:
                _logger.info("{}, from {} to {}".format(
                    param.name, original_shapes[param.name], param.shape))
        _logger.info(
            "####################channel pruning end##########################")

        final_flops = flops(pruned_program)
        pruned_flops = abs(origin_flops - final_flops) / origin_flops
        _logger.info("FLOPs before pruning: {}".format(origin_flops))
        _logger.info("FLOPs after pruning: {}. Pruned FLOPs: {}%.".format(
            final_flops, round(pruned_flops * 100, 2)))
        train_program_info.program = pruned_program

    elif strategy.startswith('asp'):
        from paddle.incubate import asp
        pruner = asp
        excluded_params_name = []
        if config['prune_params_name'] is None:
            config['prune_params_name'] = _get_asp_prune_params(
                train_program_info.program)

        for param in train_program_info.program.global_block().all_parameters():
            if config['prune_params_name'] is not None:
                if param.name not in config['prune_params_name']:
                    excluded_params_name.append(param.name)
                else:
                    pruner.add_supported_layer(param.name)
            if "teacher_" in param.name:
                excluded_params_name.append(param.name)
        pruner.set_excluded_layers(
            main_program=train_program_info.program,
            param_names=excluded_params_name)
    elif strategy.startswith('transformer_prune'):
        from .transformer_pruner import TransformerPruner
        assert eval_dataloader is not None, "transformer_pruner must set eval_dataloader"
        label_info = _get_label_info(eval_dataloader,
                                     train_program_info.feed_target_names)
        assert len(label_info) != 0, \
            "maybe something wrong in get label name from eval_dataloader, please check your eval_dataloader"
        pruner = TransformerPruner(
            executor,
            place,
            train_program_info.program,
            patterns,
            label_info,
            width_mult=(1.0 - config['pruned_ratio']),
            dataloader=eval_dataloader,
            fetch_targets=train_program_info.fetch_targets)
        pruned_program = pruner.prune()
        train_program_info.program = pruned_program
    else:
        raise NotImplementedError(
            "prune_algo must be choice in [\"prune\", \"asp\"], {} is not support".
            format(config['prune_algo']))

    return pruner, train_program_info


def remove_unused_var_nodes(program):
    '''
    This function is called before saving the sparse model to remove redundant nodes.
    Args:
        program(paddle.static.Program): The sparse model to be saved.
    Returns:
        program(paddle.static.Program): The sparse model.
    '''
    from paddle.framework import core
    from paddle.framework import IrGraph
    graph = IrGraph(core.Graph(program.desc), for_test=True)
    removed_nodes = set()
    ops = graph.all_op_nodes()
    for op_node in ops:
        for input_node in op_node.inputs:
            if '_mask' in input_node.name():
                removed_nodes.add(op_node)
    graph.safe_remove_nodes(removed_nodes)
    program = graph.to_program()
    return program
