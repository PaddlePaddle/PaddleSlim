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
import copy
import numpy as np
import copy
import inspect
import shutil
from time import gmtime, strftime
import platform
import paddle
import itertools
import paddle.distributed.fleet as fleet
from ..quant.quanter import convert, quant_post
from ..common.recover_program import recover_inference_program
from ..common import get_logger
from ..common.patterns import get_patterns
from ..analysis import TableLatencyPredictor
from .create_compressed_program import build_distill_program, build_quant_program, build_prune_program, remove_unused_var_nodes
from .strategy_config import TrainConfig, ProgramInfo, merge_config
from .auto_strategy import prepare_strategy, get_final_quant_config, create_strategy_config, create_train_config
from .config_helpers import load_config, extract_strategy_config, extract_train_config
from .utils.predict import with_variable_shape
from .utils import get_feed_vars, wrap_dataloader, load_inference_model

_logger = get_logger(__name__, level=logging.INFO)

try:
    if platform.system().lower() == 'linux':
        from ..quant import post_quant_hpo
except Exception as e:
    _logger.warning(e)


class AutoCompression:
    def __init__(self,
                 model_dir,
                 model_filename,
                 params_filename,
                 save_dir,
                 train_dataloader,
                 config=None,
                 input_shapes=None,
                 target_speedup=None,
                 eval_callback=None,
                 eval_dataloader=None,
                 deploy_hardware='gpu'):
        """
        Compress inference model automatically.

        Args:
            model_dir(str): The path of inference model that will be compressed, and
                the model and params that saved by ``paddle.static.save_inference_model``
                are under the path.
            model_filename(str):  The name of model file. 
            params_filename(str): The name of params file.
            save_dir(str): The path to save compressed model. The models in this directory will be overwrited
                after calling 'compress()' function.
            train_data_loader(Python Generator, Paddle.io.DataLoader): The
                Generator or Dataloader provides train data, and it could
                return a batch every time.
            input_shapes(dict|tuple|list): It is used when the model has implicit dimensions except batch size. 
                If it is a dict, the key is the name of input and the value is the shape. 
                Given the input shape of input "X" is [-1, 3, -1, -1] which means the batch size, hight
                and width is variable. And the input_shapes can be set {"X": [-1, 3, 512, 512]}.
                If it is a list or tuple, the number of model's inputs should be 1. And the shape of input
                will be set input_shapes. None means keeping the original shapes, then
                the compression strategies searching may be skipped. Default: None.
            train_config(dict, optional): The train config in the compression process, the key can 
                reference `<https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L103>`_ . 
                Only one strategy(quant_post with hyperparameter optimization) can set train_config 
                to None. Default: None. 
            strategy_config(dict, list(dict), optional): The strategy config. You can set single config to get multi-strategy config, such as
                1. set ``Quantization`` and ``Distillation`` to get quant_aware and distillation compress config.
                    The Quantization config can reference `https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L24`_ .
                    The Distillation config can reference `https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L39`_ .
                2. set ``Quantization`` and ``HyperParameterOptimization`` to get quant_post and hyperparameter optimization compress config.
                    The Quantization config can reference `https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L24`_ .
                    The HyperParameterOptimization config can reference `https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L73`_ .
                3. set ``ChannelPrune`` and ``Distillation`` to get channel prune and distillation compress config.
                    The ChannelPrune config can reference `https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L82`_ .
                    The Distillation config can reference `https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L39`_ .
                4. set ``ASPPrune`` and ``Distillation`` to get asp prune and distillation compress config.
                    The ASPPrune config can reference `https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L82`_ .
                    The Distillation config can reference `https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L39`_ .
                5. set ``TransformerPrune`` and ``Distillation`` to get transformer prune and distillation compress config.
                    The TransformerPrune config can reference `https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L82`_ .
                    The Distillation config can reference `https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L39`_ .
                6. set ``UnstructurePrune`` and ``Distillation`` to get unstructureprune and distillation compress config.
                    The UnstructurePrune config can reference `https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L91`_ .
                    The Distillation config can reference `https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L39`_ .
                7. set ``Distillation`` to use one teacher modol to distillation student model.
                    The Distillation config can reference `https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L39`_ .
                8. set ``MultiTeacherDistillation`` to use multi-teacher to distillation student model.
                    The MultiTeacherDistillation config can reference `https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/auto_compression/strategy_config.py#L56`_ .

                If set to None, will choose a strategy automatically. Default: None.
            target_speedup(float, optional): target speedup ratio by the way of auto compress. Default: None.
            eval_callback(function, optional): eval function, define by yourself to return the metric of the inference program, can be used to judge the metric of compressed model. The documents of how to write eval function is `https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/static/auto-compression/custom_function.rst`_ . ``eval_callback`` and ``eval_dataloader`` cannot be None at the same time. Dafault: None.
            eval_dataloader(paddle.io.Dataloader, optional):  The Generator or Dataloader provides eval data, and it could
                 return a batch every time. If eval_dataloader is None, will take first 5000 sample from train_dataloader 
                 as eval_dataloader, and the metric of eval_dataloader for reference only. Dafault: None.
            deploy_hardware(str, optional): The hardware you want to deploy. Default: 'gpu'.
        """
        self.model_dir = model_dir.rstrip('/')

        if model_filename == 'None':
            model_filename = None
        self.model_filename = model_filename
        if params_filename == 'None':
            params_filename = None
        self.params_filename = params_filename

        if params_filename is None and model_filename is not None:
            raise NotImplementedError(
                "NOT SUPPORT parameters saved in separate files. Please convert it to single binary file first."
            )

        self.final_dir = save_dir
        if not os.path.exists(self.final_dir):
            os.makedirs(self.final_dir)

        # load config
        if isinstance(config, str):
            config = load_config(config)
            self.train_config = extract_train_config(config)
        elif isinstance(config, dict):
            if 'TrainConfig' in config:
                self.train_config = TrainConfig(**config.pop('TrainConfig'))
            else:
                self.train_config = None
        self.strategy_config = extract_strategy_config(config)

        # prepare dataloader
        self.feed_vars = get_feed_vars(self.model_dir, model_filename,
                                       params_filename)
        self.train_dataloader = wrap_dataloader(train_dataloader,
                                                self.feed_vars)
        self.eval_dataloader = wrap_dataloader(eval_dataloader, self.feed_vars)
        if eval_dataloader is None:
            eval_dataloader = self._get_eval_dataloader(self.train_dataloader)

        self.target_speedup = target_speedup
        self.eval_function = eval_callback
        self.deploy_hardware = deploy_hardware

        paddle.enable_static()
        self._exe, self._places = self._prepare_envs()
        self.model_type = self._get_model_type(self._exe, self.model_dir,
                                               model_filename, params_filename)

        if self.train_config is not None and self.train_config.use_fleet:
            fleet.init(is_collective=True)

        if with_variable_shape(
                self.model_dir,
                model_filename=model_filename,
                params_filename=params_filename) and input_shapes is not None:

            infer_shape_model = self.create_tmp_dir(
                self.final_dir, prefix="infer_shape_model_")
            self._infer_shape(self.model_dir, self.model_filename,
                              self.params_filename, input_shapes,
                              infer_shape_model)
            self.model_dir = infer_shape_model
            self.model_filename = "infered_shape.pdmodel"
            self.params_filename = "infered_shape.pdiparams"

        if self.strategy_config is None:
            strategy_config = prepare_strategy(
                self._exe, self._places, self.model_dir, self.model_filename,
                self.params_filename, self.target_speedup, self.deploy_hardware,
                self.model_type)
            self.strategy_config = strategy_config
        elif isinstance(self.strategy_config, dict):
            self.strategy_config = [self.strategy_config]
        elif isinstance(self.strategy_config, str):
            strategy_config = create_strategy_config(self.strategy_config,
                                                     self.model_type)

        self._strategy, self._config = self._prepare_strategy(
            self.strategy_config)

        self.train_config = self._get_final_train_config(
            self.train_config, self._strategy, self.model_type)
        _logger.info(f"Selected strategies: {self._strategy}")

    def _get_final_train_config(self, train_config, strategy_config,
                                model_type):
        # If train_config is None, set default train_config
        if train_config is None:
            train_config = create_train_config(strategy_config, model_type)

        train_configs = [train_config]
        for idx in range(1, len(self._strategy)):
            if 'qat' in self._strategy[idx] or 'ptq' in self._strategy[idx]:
                ### If compress strategy more than one, the TrainConfig in the yaml only used in prune.
                ### The TrainConfig for quantization is extrapolate from above.
                tmp_train_config = copy.deepcopy(train_config.__dict__)
                ### the epoch, train_iter, learning rate of quant is 10% of the prune compress
                if self.model_type != 'transformer':
                    tmp_train_config['epochs'] = max(
                        int(train_config.epochs * 0.1), 1)
                if train_config.train_iter is not None:
                    tmp_train_config['train_iter'] = int(
                        train_config.train_iter * 0.1)
                if isinstance(train_config.learning_rate, float):
                    tmp_train_config[
                        'learning_rate'] = train_config.learning_rate * 0.1
                else:
                    if 'learning_rate' in train_config.learning_rate:
                        tmp_train_config['learning_rate'][
                            'learning_rate'] = train_config.learning_rate[
                                'learning_rate'] * 0.1
                    else:  ### learning rate decay is PiecewiseDecay
                        tmp_train_config['learning_rate']['values'] = list(
                            map(lambda x: x * 0.1, train_config.learning_rate[
                                'values']))
                train_cfg = TrainConfig(**tmp_train_config)
            else:
                tmp_train_config = copy.deepcopy(train_config.__dict__)
                train_cfg = TrainConfig(**tmp_train_config)

            train_configs.append(train_cfg)
        return train_configs

    def _infer_shape(self, model_dir, model_filename, params_filename,
                     input_shapes, save_path):
        assert type(input_shapes) in [
            dict, list, tuple
        ], f'Type of input_shapes should be in [dict, tuple or list] but got {type(input_shapes)}.'
        paddle.enable_static()
        exe = paddle.static.Executor(paddle.CPUPlace())
        [inference_program, feed_target_names,
         fetch_targets] = (load_inference_model(model_dir, exe, model_filename,
                                                params_filename))

        if type(input_shapes) in [list, tuple]:
            assert len(
                feed_target_names
            ) == 1, f"The number of model's inputs should be 1 but got {feed_target_names}."
            input_shapes = {feed_target_names[0]: input_shapes}

        feed_vars = []
        for var_ in inference_program.list_vars():
            if var_.name in feed_target_names:
                feed_vars.append(var_)
                var_.desc.set_shape(input_shapes[var_.name])

        for block in inference_program.blocks:
            for op in block.ops:
                if op.type not in ["feed", "fetch"]:
                    op.desc.infer_shape(block.desc)

        save_path = os.path.join(save_path, "infered_shape")
        os.makedirs(save_path)
        paddle.static.save_inference_model(
            save_path, feed_vars, fetch_targets, exe, program=inference_program)
        _logger.info(f"Saved model infered shape to {save_path}")

    @property
    def deploy_hardware(self):
        return self._deploy_hardware

    @deploy_hardware.setter
    def deploy_hardware(self, value):
        supported_hardware = TableLatencyPredictor.hardware_list + [
            'gpu',  # nvidia gpu
            "cpu",  # intel cpu
        ]
        if value is not None:
            # Fail-fast when deploy hardware is set explicitly
            assert (
                value in supported_hardware
            ), f"Hardware should be in supported list {supported_hardware} but got {value}. Or you can set deploy_hardware None."
        self._deploy_hardware = value

    def _get_eval_dataloader(self, train_dataloader):
        def _gen():
            len_loader = len(list(train_dataloader()))
            ### max eval_dataloader is 5000 if use train_dataloader as eval_dataloader
            slice_len = min(5000, len_loader)
            ret = list(itertools.islice(train_dataloader(), slice_len))
            for i in ret:
                yield i

        return _gen

    def _prepare_envs(self):
        devices = paddle.device.get_device().split(':')[0]
        places = paddle.device._convert_to_place(devices)
        _logger.info(f"devices: {devices}")
        exe = paddle.static.Executor(places)
        return exe, places

    def _get_model_type(self, exe, model_dir, model_filename, params_filename):
        [inference_program, _, _]= (load_inference_model( \
            model_dir, \
            model_filename=model_filename, params_filename=params_filename,
            executor=exe))
        _, _, model_type = get_patterns(inference_program)
        if self.model_filename is None:
            new_model_filename = '__new_model__'
        else:
            new_model_filename = 'new_' + self.model_filename
        program_bytes = inference_program._remove_training_info(
            clip_extra=False).desc.serialize_to_string()
        with open(os.path.join(self.model_dir, new_model_filename), "wb") as f:
            f.write(program_bytes)
        shutil.move(
            os.path.join(self.model_dir, new_model_filename),
            os.path.join(self.model_dir, self.model_filename))
        _logger.info(f"Detect model type: {model_type}")
        return model_type

    def _prepare_strategy(self, strategy_config):
        if not isinstance(strategy_config, list):
            strategy_config = list(list(strategy_config))

        strategy = []
        config = []
        for strategy_c in strategy_config:
            quant_config = strategy_c.get("Quantization", None)
            hpo_config = strategy_c.get("HyperParameterOptimization", None)
            prune_config = strategy_c.get("ChannelPrune", None)
            asp_config = strategy_c.get("ASPPrune", None)
            transformer_prune_config = strategy_c.get("TransformerPrune", None)
            unstructure_prune_config = strategy_c.get("UnstructurePrune", None)
            single_teacher_distill_config = strategy_c.get("Distillation", None)
            if single_teacher_distill_config is not None and single_teacher_distill_config.teacher_model_dir is None:
                single_teacher_distill_config.teacher_model_dir = self.model_dir
                single_teacher_distill_config.teacher_model_filename = self.model_filename
                single_teacher_distill_config.teacher_params_filename = self.params_filename

            multi_teacher_distill_config = strategy_c.get(
                "MultiTeacherDistillation", None)

            assert (single_teacher_distill_config is None) or (multi_teacher_distill_config is None), \
                "Distillation and MultiTeacherDistillation cannot be set at the same time."
            self._distill_config = single_teacher_distill_config if \
                   single_teacher_distill_config is not None else \
                   multi_teacher_distill_config

            only_distillation = True

            ### case1: prune_config & distill config
            if prune_config is not None and self._distill_config is not None:
                only_distillation = False
                strategy.append('channel_prune_dis')
                config.append(merge_config(prune_config, self._distill_config))

            ### case2: asp_config & distill config
            if asp_config is not None and self._distill_config is not None:
                only_distillation = False
                strategy.append('asp_prune_dis')
                config.append(merge_config(asp_config, self._distill_config))

            ### case3: transformer_prune_config & distill config
            if transformer_prune_config is not None and self._distill_config is not None:
                only_distillation = False
                strategy.append('transformer_prune_dis')
                config.append(
                    merge_config(transformer_prune_config,
                                 self._distill_config))

            ### case4: unstructure_config & distill config
            if unstructure_prune_config is not None and self._distill_config is not None:
                only_distillation = False
                strategy.append('unstructure_prune_dis')
                config.append(
                    merge_config(unstructure_prune_config,
                                 self._distill_config))

            ### case5: quant_config & hpo_config ==> PTQ & HPO
            if quant_config is not None and hpo_config is not None:
                only_distillation = False
                strategy.append('ptq_hpo')
                config.append(merge_config(quant_config, hpo_config))

            ### case6: quant_config & distill config ==> QAT & Distill
            if quant_config is not None and self._distill_config is not None:
                only_distillation = False
                strategy.append('qat_dis')
                config.append(merge_config(quant_config, self._distill_config))

            ### case7: distill_config
            if only_distillation == True and self._distill_config is not None:
                if single_teacher_distill_config is not None:
                    strategy.append('single_teacher_dis')
                    config.append(single_teacher_distill_config)
                else:
                    strategy.append('multi_teacher_dis')
                    config.append(multi_teacher_distill_config)

        ### NOTE: keep quantation in the last step
        idx = -1
        if 'qat_dis' in strategy and strategy.index('qat_dis') != (
                len(strategy) - 1):
            idx = strategy.index('qat_dis')
        elif 'ptq_hpo' in strategy and strategy.index('ptq_hpo') != (
                len(strategy) - 1):
            idx = strategy.index('ptq_hpo')

        if idx != -1:
            strategy = strategy[:idx] + strategy[idx + 1:] + [strategy[idx]]
            config = config[:idx] + config[idx + 1:] + [config[idx]]

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
        if train_config.asp_config is not None:
            strategy.asp = True
        return strategy

    def _prepare_program(self, program, feed_target_names, fetch_targets,
                         patterns, default_distill_node_pair, strategy, config,
                         train_config):
        train_program = recover_inference_program(program)
        startup_program = paddle.static.Program()
        train_program_info = ProgramInfo(startup_program, train_program,
                                         feed_target_names, fetch_targets)

        config_dict = config.__dict__
        if "prune_strategy" in config_dict and config_dict[
                "prune_strategy"] == "gmp" and config_dict[
                    'gmp_config'] is None:
            _logger.info(
                "Calculating the iterations per epoch……(It will take some time)")
            # NOTE:XXX: This way of calculating the iters needs to be improved.
            if train_config.epochs:
                iters_per_epoch = len(list(self.train_dataloader()))
                total_iters = train_config.epochs * iters_per_epoch
            elif train_config.train_iter:
                total_iters = train_config.train_iter
            else:
                raise RuntimeError(
                    'train_config must has `epochs` or `train_iter` field.')
            config_dict['gmp_config'] = {
                'stable_iterations': 0,
                'pruning_iterations': 0.45 * total_iters,
                'tunning_iterations': 0.45 * total_iters,
                'resume_iteration': -1,
                'pruning_steps': 100,
                'initial_ratio': 0.15,
            }
        ### add prune program
        self._pruner = None
        if 'prune' in strategy:
            self._pruner, train_program_info = build_prune_program(
                self._exe, self._places, config_dict, train_program_info,
                strategy, patterns, self.eval_dataloader)

        if train_config.use_fleet:
            dist_strategy = _prepare_fleet_strategy(train_config)
        else:
            dist_strategy = None

        ### add distill program
        if 'dis' in strategy:
            train_program_info, test_program_info = build_distill_program(
                self._exe,
                self._places,
                config_dict,
                train_config.__dict__,
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
        if train_config.sparse_model:
            from ..prune.unstructured_pruner import UnstructuredPruner
            # NOTE: The initialization parameter of this pruner doesn't work, it is only used to call the 'set_static_masks' function
            self._pruner = UnstructuredPruner(
                train_program_info.program,
                mode='ratio',
                ratio=0.75,
                prune_params_type='conv1x1_only',
                place=self._places)
            self._pruner.set_static_masks()  # Fixed model sparsity

        self._exe.run(train_program_info.startup_program)

        if (not train_config.use_fleet) and train_config.amp_config is not None:
            if hasattr(
                    train_config.amp_config,
                    'use_pure_fp16') and train_config.amp_config.use_pure_fp16:
                train_program_info.optimizer.amp_init(
                    self._places, scope=paddle.static.global_scope())

        if 'asp' in strategy:
            ### prune weight in scope
            self._pruner.prune_model(train_program_info.program)

        if not train_config.use_fleet:
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

    def create_tmp_dir(self, base_dir, prefix="tmp"):
        # create a new temp directory in final dir
        s_datetime = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
        tmp_base_name = "_".join([prefix, str(os.getpid()), s_datetime])
        tmp_dir = os.path.join(base_dir, tmp_base_name)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        return tmp_dir

    def compress(self):
        assert len(self._strategy) > 0
        self.tmp_dir = self.create_tmp_dir(self.final_dir)
        strategy = None
        config = None
        train_config = None
        strategy_idx = None
        for strategy_idx, (
                strategy, config, train_config
        ) in enumerate(zip(self._strategy, self._config, self.train_config)):
            self.single_strategy_compress(strategy, config, strategy_idx,
                                          train_config)

        if strategy == 'ptq_hpo' and config.max_quant_count == 1 and platform.system(
        ).lower() == 'linux':
            ptq_loss = post_quant_hpo.g_min_emd_loss

            final_quant_config = get_final_quant_config(ptq_loss,
                                                        self.model_type)
            if final_quant_config is not None:
                quant_strategy, quant_config = self._prepare_strategy(
                    final_quant_config)
                self.single_strategy_compress(quant_strategy[0],
                                              quant_config[0], strategy_idx,
                                              train_config)
        tmp_model_path = os.path.join(
            self.tmp_dir, 'strategy_{}'.format(str(strategy_idx + 1)))
        final_model_path = os.path.join(self.final_dir)
        if paddle.distributed.get_rank() == 0:
            for _file in os.listdir(tmp_model_path):
                _file_path = os.path.join(tmp_model_path, _file)
                if os.path.isfile(_file_path):
                    shutil.copy(_file_path, final_model_path)
            shutil.rmtree(self.tmp_dir)
            _logger.info(
                "==> The ACT compression has been completed and the final model is saved in `{}`".
                format(final_model_path))
        os._exit(0)

    def single_strategy_compress(self, strategy, config, strategy_idx,
                                 train_config):
        # start compress, including train/eval model
        # TODO: add the emd loss of evaluation model.
        if strategy == 'quant_post':
            quant_post(
                self._exe,
                model_dir=self.model_dir,
                quantize_model_path=os.path.join(
                    self.tmp_dir, 'strategy_{}'.format(str(strategy_idx + 1))),
                data_loader=self.train_dataloader,
                model_filename=self.model_filename,
                params_filename=self.params_filename,
                save_model_filename=self.model_filename,
                save_params_filename=self.params_filename,
                batch_size=1,
                batch_nums=config.batch_num,
                algo=config.ptq_algo,
                round_type='round',
                bias_correct=config.bias_correct,
                hist_percent=config.hist_percent,
                quantizable_op_type=config.quantize_op_types,
                is_full_quantize=config.is_full_quantize,
                weight_bits=config.weight_bits,
                activation_bits=config.activation_bits,
                activation_quantize_type='range_abs_max',
                weight_quantize_type=config.weight_quantize_type,
                onnx_format=False)

        elif strategy == 'ptq_hpo':
            if platform.system().lower() != 'linux':
                raise NotImplementedError(
                    "post-quant-hpo is not support in system other than linux")

            post_quant_hpo.quant_post_hpo(
                self._exe,
                self._places,
                model_dir=self.model_dir,
                quantize_model_path=os.path.join(
                    self.tmp_dir, 'strategy_{}'.format(str(strategy_idx + 1))),
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
                    self.tmp_dir, 'strategy_{}'.format(str(strategy_idx)))

            [inference_program, feed_target_names, fetch_targets]= load_inference_model( \
                model_dir, \
                model_filename=self.model_filename, params_filename=self.params_filename,
                executor=self._exe)

            ### used to check whether the dataloader is right
            self.metric_before_compressed = None
            if self.eval_function is not None and train_config.origin_metric is not None:
                _logger.info("start to test metric before compress")
                metric = self.eval_function(self._exe, inference_program,
                                            feed_target_names, fetch_targets)
                _logger.info("metric of compressed model is: {}".format(metric))
                buf = 0.05
                if metric < (float(train_config.origin_metric) - buf) or \
                        metric > (float(train_config.origin_metric) + buf):
                    raise RuntimeError("target metric of pretrained model is {}, \
                          but now is {}, Please check the format of evaluation dataset \
                          or check the origin_metric in train_config"
                                                                     .format(\
                          train_config.origin_metric, metric))
                self.metric_before_compressed = metric

            patterns, default_distill_node_pair, _ = get_patterns(
                inference_program)

            train_program_info, test_program_info = self._prepare_program(
                inference_program, feed_target_names, fetch_targets, patterns,
                default_distill_node_pair, strategy, config, train_config)
            if 'unstructure' in self._strategy:
                test_program_info.program._program = remove_unused_var_nodes(
                    test_program_info.program._program)
            test_program_info = self._start_train(
                train_program_info, test_program_info, strategy, train_config)
            self._save_model(test_program_info, strategy, strategy_idx)

    def _start_train(self, train_program_info, test_program_info, strategy,
                     train_config):
        best_metric = -1.0
        total_epochs = train_config.epochs if train_config.epochs else 100
        total_train_iter = 0
        for epoch_id in range(total_epochs):
            for batch_id, data in enumerate(self.train_dataloader()):
                np_probs_float, = self._exe.run(train_program_info.program, \
                    feed=data, \
                    fetch_list=train_program_info.fetch_targets)
                if not isinstance(train_program_info.learning_rate, float):
                    train_program_info.learning_rate.step()
                if 'unstructure' in strategy:
                    self._pruner.step()

                if train_config.logging_iter is None:
                    logging_iter = 10
                else:
                    logging_iter = train_config.logging_iter
                if batch_id % int(logging_iter) == 0:
                    _logger.info(
                        "Total iter: {}, epoch: {}, batch: {}, loss: {}".format(
                            total_train_iter, epoch_id, batch_id,
                            np_probs_float))
                total_train_iter += 1
                if total_train_iter % int(
                        train_config.eval_iter) == 0 and total_train_iter != 0:
                    if self.eval_function is not None:

                        # GMP pruner step 3: update params before summrizing sparsity, saving model or evaluation. 
                        if 'unstructure' in strategy:
                            self._pruner.update_params()

                        metric = self.eval_function(
                            self._exe, test_program_info.program,
                            test_program_info.feed_target_names,
                            test_program_info.fetch_targets)

                        if metric > best_metric:
                            paddle.static.save(
                                program=test_program_info.program._program,
                                model_path=os.path.join(self.tmp_dir,
                                                        'best_model'))
                            best_metric = metric
                            _logger.info(
                                "epoch: {} metric of compressed model is: {:.6f}, best metric of compressed model is {:.6f}".
                                format(epoch_id, metric, best_metric))
                            if self.metric_before_compressed is not None and float(
                                    abs(best_metric -
                                        self.metric_before_compressed)
                            ) / self.metric_before_compressed <= 0.005:
                                break
                        else:
                            _logger.info(
                                "epoch: {} metric of compressed model is: {:.6f}, best metric of compressed model is {:.6f}".
                                format(epoch_id, metric, best_metric))
                        if train_config.target_metric is not None:
                            if metric > float(train_config.target_metric):
                                break

                    else:
                        _logger.warning(
                            "Not set eval function, so unable to test accuracy performance."
                        )
                if train_config.train_iter and total_train_iter >= train_config.train_iter:
                    epoch_id = total_epochs
                    break

        if 'unstructure' in self._strategy or train_config.sparse_model:
            self._pruner.update_params()

        return test_program_info

    def _save_model(self, test_program_info, strategy, strategy_idx):
        test_program = test_program_info.program._program if isinstance(
            test_program_info.program,
            paddle.static.CompiledProgram) else test_program_info.program

        if os.path.exists(os.path.join(self.tmp_dir, 'best_model.pdparams')):
            paddle.static.load(test_program,
                               os.path.join(self.tmp_dir, 'best_model'))
            os.remove(os.path.join(self.tmp_dir, 'best_model.pdmodel'))
            os.remove(os.path.join(self.tmp_dir, 'best_model.pdopt'))
            os.remove(os.path.join(self.tmp_dir, 'best_model.pdparams'))

        if 'qat' in strategy:
            test_program, int8_program = convert(test_program, self._places, self._quant_config, \
                                          scope=paddle.static.global_scope(), \
                                          save_int8=True)

        model_dir = os.path.join(self.tmp_dir,
                                 'strategy_{}'.format(str(strategy_idx + 1)))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        feed_vars = [
            test_program.global_block().var(name)
            for name in test_program_info.feed_target_names
        ]

        model_name = None
        if self.model_filename is None:
            model_name = "model"
        elif self.model_filename.endswith(".pdmodel"):
            model_name = self.model_filename.rsplit(".", 1)[0]
        else:
            model_name = self.model_filename

        path_prefix = os.path.join(model_dir, model_name)
        paddle.static.save_inference_model(
            path_prefix=path_prefix,
            feed_vars=feed_vars,
            fetch_vars=test_program_info.fetch_targets,
            executor=self._exe,
            program=test_program)
