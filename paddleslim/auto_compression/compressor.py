#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import os
import sys
import numpy as np
import inspect
import shutil
from collections import namedtuple, Iterable
import platform
import paddle
import paddle.distributed.fleet as fleet
if platform.system().lower() == 'linux':
    from ..quant import quant_post_hpo
from ..quant.quanter import convert
from ..common.recover_program import recover_inference_program
from ..common import get_logger
from ..analysis import TableLatencyPredictor
from .create_compressed_program import build_distill_program, build_quant_program, build_prune_program
from .strategy_config import ProgramInfo, merge_config
from .patterns import get_patterns
from .auto_strategy import auto_prepare_strategy, get_final_quant_config

_logger = get_logger(__name__, level=logging.INFO)


class AutoCompression:
    def __init__(self,
                 model_dir,
                 model_filename,
                 params_filename,
                 save_dir,
                 train_config,
                 train_dataloader,
                 strategy_config=None,
                 target_speedup=None,
                 eval_callback=None,
                 eval_dataloader=None,
                 devices='gpu',
                 model_type=None,
                 deploy_hardware='gpu'):
        ### model_dir(str): 模型路径
        ### model_filename(str): 模型文件名称
        ### params_filename(str): 参数文件名称
        ### save_dir(str): 压缩后模型保存的路径
        ### strategy_config(dict[dict]): 压缩策略配置, 包括量化配置、蒸馏配置
        ### train_config(dict): 训练配置
        ### train_dataloader(paddle.nn.Dataloader): 训练数据dataloader
        ### eval_callback(function，paddle.nn.Dataloader): eval回调函数，和测试数据之间必须传入一个，如果传入回调函数，则使用回调函数判断模型训练情况。callback传入predict结果（paddle的tensor），默认：None。
        self.model_dir = model_dir
        self.model_filename = model_filename
        self.params_filename = params_filename
        self.save_dir = save_dir
        self.strategy_config = strategy_config
        self.train_config = train_config
        self.train_dataloader = train_dataloader
        self.target_speedup = target_speedup
        self.eval_function = eval_callback
        self.eval_dataloader = eval_dataloader
        self.model_type = model_type

        if deploy_hardware in TableLatencyPredictor.hardware_list:
            self.deploy_hardware = deploy_hardware
        else:
            self.deploy_hardware = None

        paddle.enable_static()
        if self.train_config is not None and self.train_config.use_fleet:
            fleet.init(is_collective=True)

        if self.strategy_config is None:
            strategy_config = auto_prepare_strategy(
                self.model_dir, self.model_filename, self.params_filename,
                self.target_speedup, self.deploy_hardware, self.model_type)
            self.strategy_config = strategy_config
        elif isinstance(self.strategy_config, dict):
            self.strategy_config = [self.strategy_config]

        self._strategy, self._config = self._prepare_strategy(
            self.strategy_config)

        self._exe, self._places = self._prepare_envs(devices)

    def _prepare_envs(self, devices):
        places = paddle.device._convert_to_place(devices)
        exe = paddle.static.Executor(places)
        return exe, places

    def _prepare_strategy(self, strategy_config):
        if not isinstance(strategy_config, list):
            strategy_config = list(list(strategy_config))

        strategy = []
        config = []
        for strategy_c in strategy_config:
            quant_config = strategy_c.get("Quantization", None)
            hpo_config = strategy_c.get("HyperParameterOptimization", None)
            prune_config = strategy_c.get("Prune", None)
            unstructure_prune_config = strategy_c.get("UnstructurePrune", None)
            single_teacher_distill_config = strategy_c.get("Distillation", None)
            if single_teacher_distill_config is not None and single_teacher_distill_config.teacher_model_dir is None:
                single_teacher_distill_config = single_teacher_distill_config._replace(
                    teacher_model_dir=self.model_dir,
                    teacher_model_filename=self.model_filename,
                    teacher_params_filename=self.params_filename)

            multi_teacher_distill_config = strategy_c.get(
                "MultiTeacherDistillation", None)

            assert (single_teacher_distill_config is None) or (multi_teacher_distill_config is None), \
                "Distillation and MultiTeacherDistillation cannot be set at the same time."
            self._distill_config = single_teacher_distill_config if \
                   single_teacher_distill_config is not None else \
                   multi_teacher_distill_config

            ### case1: quant_config & hpo_config ==> PTQ & HPO
            if quant_config is not None and hpo_config is not None:
                strategy.append('ptq_hpo')
                config.append(merge_config(quant_config, hpo_config))

            ### case2: quant_config & distill config ==> QAT & Distill
            elif quant_config is not None and self._distill_config is not None:
                strategy.append('qat_dis')
                config.append(merge_config(quant_config, self._distill_config))

            ### case3: prune_config & distill config
            elif prune_config is not None and self._distill_config is not None:
                strategy.append('prune_dis')
                config.append(merge_config(prune_config, self._distill_config))

            ### case4: unstructure_config & distill config
            elif unstructure_prune_config is not None and self._distill_config is not None:
                strategy.append('unstructure_prune_dis')
                config.append(
                    merge_config(unstructure_prune_config,
                                 self._distill_config))

            ### case4: distill_config
            elif self._distill_config is not None:
                if single_teacher_distill_config is not None:
                    strategy.append('single_teacher_dis')
                    config.append(single_teacher_distill_config)
                else:
                    strategy.append('multi_teacher_dis')
                    config.append(multi_teacher_distill_config)

            ### case N: todo
            else:
                raise NotImplementedError(
                    "Not Implemented {} be set at the same time now".format(
                        strategy_c.keys()))

        return strategy, config

    def _prepare_fleet_strategy(train_config):
        build_strategy = paddle.static.BuildStrategy()
        exec_strategy = paddle.static.ExecutionStrategy()

        strategy = fleet.DistributedStrategy()
        strategy.build_strategy = build_strategy
        if train_config.recompute_config is not None:
            strategy.recompute = True
            strategy.recompute_configs = { ** train_config.recompute_config}
        if train_config.sharding_config is not None:
            strategy.sharding = True
            strategy.sharding_configs = { ** train_config.sharding_config}
        if train_config.amp_config is not None:
            strategy.amp = True
            strategy.amp_configs = { ** train_config.amp_config}
        return strategy

    def _prepare_program(self, program, feed_target_names, fetch_targets,
                         patterns, default_distill_node_pair, strategy, config):
        train_program = recover_inference_program(program)
        startup_program = paddle.static.Program()
        train_program_info = ProgramInfo(startup_program, train_program,
                                         feed_target_names, fetch_targets)

        config_dict = dict(config._asdict())
        ### add prune program
        self._pruner = None
        if 'prune' in strategy:
            self._pruner, train_program_info = build_prune_program(
                self._exe, self._places, config_dict, train_program_info,
                strategy, patterns, self.eval_dataloader)

        if self.train_config.use_fleet:
            dist_strategy = _prepare_fleet_strategy(self.train_config)
        else:
            dist_strategy = None

        ### add distill program
        if 'dis' in strategy:
            train_program_info, test_program_info = build_distill_program(
                self._exe,
                self._places,
                config_dict,
                self.train_config._asdict(),
                train_program_info,
                pruner=self._pruner,
                dist_strategy=dist_strategy,
                default_distill_node_pair=default_distill_node_pair)

        self._quant_config = None
        ### add quant_aware program, quant always is last step
        if 'qat' in strategy:
            train_program_info, test_program_info, self._quant_config = build_quant_program(
                self._exe, self._places, config_dict, train_program_info,
                test_program_info)

        self._exe.run(train_program_info.startup_program)

        if (not self.train_config.use_fleet
            ) and self.train_config.amp_config is not None:
            if hasattr(self.train_config.amp_config, 'use_pure_fp16'
                       ) and self.train_config.amp_config.use_pure_fp16:
                train_program_info.optimizer.amp_init(
                    self._places, scope=paddle.static.global_scope())

        if 'prune_algo' in config_dict and config_dict['prune_algo'] == 'asp':
            ### prune weight in scope
            self._pruner.prune_model(train_program_info.program)

        if not self.train_config.use_fleet:
            train_program_info = self._compiled_program(train_program_info,
                                                        strategy)
            test_program_info = self._compiled_program(test_program_info,
                                                       self._strategy)
        return train_program_info, test_program_info

    def _compiled_program(self, program_info, strategy):
        compiled_prog = paddle.static.CompiledProgram(program_info.program)
        build_strategy = paddle.static.BuildStrategy()
        exec_strategy = paddle.static.ExecutionStrategy()
        if 'qat' in strategy:
            build_strategy.memory_optimize = False
            build_strategy.enable_inplace = False
            build_strategy.fuse_all_reduce_ops = False
            build_strategy.sync_batch_norm = False

        compiled_prog = compiled_prog.with_data_parallel(
            loss_name=program_info.fetch_targets[0].name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
        program_info.program = compiled_prog
        return program_info

    def compress(self):
        for strategy_idx, (
                strategy,
                config) in enumerate(zip(self._strategy, self._config)):
            self.single_strategy_compress(strategy, config, strategy_idx)

        if strategy == 'ptq_hpo' and config.max_quant_count == 1 and platform.system(
        ).lower() == 'linux':
            ptq_loss = quant_post_hpo.g_min_emd_loss

            final_quant_config = get_final_quant_config(ptq_loss)
            quant_strategy, quant_config = self._prepare_strategy(
                final_quant_config)
            self.single_strategy_compress(quant_strategy[0], quant_config[0],
                                          strategy_idx)
        old_model_path = os.path.join(
            self.save_dir, 'strategy_{}'.format(str(strategy_idx + 1)))
        final_model_path = os.path.join(self.save_dir)
        shutil.move(old_model_path, final_model_path)
        os._exit(0)

    def single_strategy_compress(self, strategy, config, strategy_idx):
        ### start compress, including train/eval model
        if strategy == 'ptq_hpo':
            if platform.system().lower() != 'linux':
                raise NotImplementedError(
                    "post-quant-hpo is not support in system other than linux")

            quant_post_hpo.quant_post_hpo(
                self._exe,
                self._places,
                model_dir=self.model_dir,
                quantize_model_path=os.path.join(
                    self.save_dir, 'strategy_{}'.format(str(strategy_idx))),
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                eval_function=self.eval_function,
                model_filename=self.model_filename,
                params_filename=self.params_filename,
                save_model_filename=self.model_filename,
                save_params_filename=self.params_filename,
                quantizable_op_type=config.quantize_op_types,
                weight_bits=config.weight_bits,
                activation_bits=config.activation_bits,
                weight_quantize_type=config.weight_quantize_type,
                is_full_quantize=config.is_full_quantize,
                algo=config.ptq_algo,
                bias_correct=config.bias_correct,
                hist_percent=config.hist_percent,
                batch_size=[1],
                batch_num=config.batch_num,
                runcount_limit=config.max_quant_count)

        else:
            assert 'dis' in strategy, "Only support optimizer compressed model by distillation loss."

            if strategy_idx == 0:
                model_dir = self.model_dir
            else:
                model_dir = os.path.join(
                    model_dir, 'strategy_{}'.format(str(strategy_idx)))

            [inference_program, feed_target_names, fetch_targets]= paddle.fluid.io.load_inference_model( \
                dirname=model_dir, \
                model_filename=self.model_filename, params_filename=self.params_filename,
                executor=self._exe)

            ### used to check whether the dataloader is right
            if self.eval_function is not None and self.train_config.origin_metric is not None:
                _logger.info("start to test metric before compress")
                metric = self.eval_function(self._exe, inference_program,
                                            feed_target_names, fetch_targets)
                _logger.info("metric of compressed model is: {}".format(metric))
                buf = 0.05
                if metric < (float(self.train_config.origin_metric) - buf) or \
                        metric > (float(self.train_config.origin_metric) + buf):
                    raise RuntimeError("target metric of pretrained model is {}, \
                          but now is {}, Please check the format of evaluation dataset \
                          or check the origin_metric in train_config"
                                                                     .format(\
                          self.train_config.origin_metric, metric))

            patterns, default_distill_node_pair = get_patterns(
                inference_program, self.model_type)

            train_program_info, test_program_info = self._prepare_program(
                inference_program, feed_target_names, fetch_targets, patterns,
                default_distill_node_pair, strategy, config)

            test_program_info = self._start_train(train_program_info,
                                                  test_program_info, strategy)
            self._save_model(test_program_info, strategy, strategy_idx)

    def _start_train(self, train_program_info, test_program_info, strategy):
        best_metric = -1.0
        for epoch_id in range(self.train_config.epochs):
            for batch_id, data in enumerate(self.train_dataloader()):
                np_probs_float, = self._exe.run(train_program_info.program, \
                    feed=data, \
                    fetch_list=train_program_info.fetch_targets)

                if 'unstructure' in strategy:
                    self._pruner.step()

                if self.train_config.logging_iter is None:
                    logging_iter = 10
                else:
                    logging_iter = self.train_config.logging_iter
                if batch_id % int(logging_iter) == 0:
                    _logger.info("epoch: {}, batch: {}, loss: {}".format(
                        epoch_id, batch_id, np_probs_float))

                if batch_id % int(
                        self.train_config.eval_iter) == 0 and batch_id != 0:
                    if self.eval_function is not None:

                        # GMP pruner step 3: update params before summrizing sparsity, saving model or evaluation. 
                        if 'unstructure' in strategy:
                            self._pruner.update_params()

                        metric = self.eval_function(
                            self._exe, test_program_info.program,
                            test_program_info.feed_target_names,
                            test_program_info.fetch_targets)

                        _logger.info(
                            "epoch: {}, batch: {} metric of compressed model is: {}".
                            format(epoch_id, batch_id, metric))
                        if metric > best_metric:
                            paddle.static.save(
                                program=test_program_info.program._program,
                                model_path=os.path.join(self.save_dir,
                                                        'best_model'))
                            best_metric = metric
                        if self.train_config.target_metric is not None:
                            if metric > float(self.train_config.target_metric):
                                return

                    else:
                        raise NotImplementedError(
                            "Please support eval function")

                paddle.static.save(
                    program=test_program_info.program._program,
                    model_path=os.path.join(self.save_dir, 'best_model'))
        return test_program_info

    def _save_model(self, test_program_info, strategy, strategy_idx):
        test_program = test_program_info.program._program if isinstance(
            test_program_info.program,
            paddle.static.CompiledProgram) else test_program_info.program

        paddle.static.load(test_program,
                           os.path.join(self.save_dir, 'best_model'))
        os.remove(os.path.join(self.save_dir, 'best_model.pdmodel'))
        os.remove(os.path.join(self.save_dir, 'best_model.pdopt'))
        os.remove(os.path.join(self.save_dir, 'best_model.pdparams'))

        if 'qat' in strategy:
            float_program, int8_program = convert(test_program_info.program._program, self._places, self._quant_config, \
                                          scope=paddle.static.global_scope(), \
                                          save_int8=True)
            test_program_info.program = float_program

        model_dir = os.path.join(self.save_dir,
                                 'strategy_{}'.format(str(strategy_idx + 1)))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        paddle.fluid.io.save_inference_model(
            dirname=str(model_dir),
            feeded_var_names=test_program_info.feed_target_names,
            target_vars=test_program_info.fetch_targets,
            executor=self._exe,
            main_program=test_program)
