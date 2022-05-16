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

import os
import logging
import platform
from ..common import get_logger
from .utils.predict import predict_compressed_model
from .strategy_config import *

_logger = get_logger(__name__, level=logging.INFO)

__all__ = [
    "prepare_strategy", "create_strategy_config", "get_final_quant_config"
]

### config tester to test the loss of quant_post
hpo_config_tester = {
    "ptq_algo": ["avg", "mse", "KL"],
    "weight_quantize_type": ['channel_wise_abs_max', 'abs_max'],
    "bias_correct": [False],
    "batch_num": [2, 3],
    "max_quant_count": 1,
}

### default hpo config
default_hpo_config = {
    "ptq_algo": ["KL", "hist", "avg", "mse"],
    "weight_quantize_type": ['channel_wise_abs_max', 'abs_max'],
    "bias_correct": [True, False],
    "hist_percent": [0.98, 0.999],
    "batch_num": [10, 30],
    "max_quant_count": 20,
}

### default quant config, can be used by ptq&hpo and qat&distillation
default_quant_config = {
    'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul', 'matmul'],
    'weight_bits': 8,
    'activation_bits': 8
}

EXPERIENCE_STRATEGY_WITHOUT_LOSS = [
    'sparse_0.75_fp32', 'prune_0.3_fp32', 'origin_int8', 'sparse_0.75_int8',
    'prune_0.3_int8'
]
MAGIC_SPARSE_RATIO = 0.75
### TODO: 0.03 threshold maybe not suitable, need to check
MAGIC_EMD_DISTANCE = 0.03

DEFAULT_TRANSFORMER_STRATEGY = 'prune_0.25_int8'
DEFAULT_STRATEGY = 'origin_int8'
DEFAULT_QUANT_SPEEDUP = 0.7


def create_strategy_config(strategy_str, model_type):
    """ create config according to string"""
    tmp_s = strategy_str.split('_')
    configs = []

    dis_config = Distillation()
    if len(tmp_s) == 3:
        ### TODO(ceci3): choose prune algo automatically
        if 'prune' in tmp_s[0]:
            ### default prune config
            default_prune_config = {
                'pruned_ratio': float(tmp_s[1]),
                'prune_algo': 'prune',
                'criterion': 'l1_norm'
            }
        else:
            ### default unstruture prune config
            default_prune_config = {
                'prune_strategy':
                'gmp',  ### default unstruture prune strategy is gmp
                'prune_mode': 'ratio',
                'pruned_ratio': float(tmp_s[1]),
                'local_sparsity': True,
                'prune_params_type': 'conv1x1_only'
            }
        tmp_s[0] = tmp_s[0].replace('prune', 'Prune')
        tmp_s[0] = tmp_s[0].replace('sparse', 'UnstructurePrune')
        if model_type == 'transformer' and tmp_s[0] == 'Prune':
            default_prune_config['prune_algo'] = 'transformer_pruner'
        prune_config = eval(tmp_s[0])(**default_prune_config)
        configs.append({tmp_s[0]: prune_config, 'Distillation': dis_config})

    ### TODO(ceci3): support skip some layer and full quant
    if tmp_s[-1] == 'int8':
        ### only platform is linux can use smac to do hyperparameter optimization
        ### choose quant_aware to do quantization in other platform
        if platform.system().lower() == 'linux':
            quant_config = Quantization(**default_quant_config)
            hpo_config = HyperParameterOptimization(**hpo_config_tester)
            configs.append({
                'Quantization': quant_config,
                'HyperParameterOptimization': hpo_config
            })
        else:
            quant_config = Quantization(**default_quant_config)
            dis_config = Distillation()
            configs.append({
                'Quantization': quant_config,
                'Distillation': dis_config
            })

    return configs


def prepare_strategy(model_dir,
                     model_filename,
                     params_filename,
                     target_speedup=None,
                     deploy_hardware=None,
                     model_type=None):
    """ prepare compression config automatically """
    final_strategy = None

    ### use hardware latency tabel if support
    if deploy_hardware is not None:
        compressed_time_dict = predict_compressed_model(
            model_dir,
            model_filename,
            params_filename,
            hardware=deploy_hardware)

        baseline = compressed_time_dict['origin_fp32']
        speedup_ratio = {}
        for strategy, latency in compressed_time_dict.items():
            speedup_ratio[strategy] = 1.0 - float(latency) / baseline

        sorted_speedup_ratio = sorted(speedup_ratio.items(), key=lambda x: x[1])

        ### if target speedup is None, choose strategy by experience.
        if target_speedup is None:
            max_speedup = -1.0
            for s in EXPERIENCE_STRATEGY_WITHOUT_LOSS:
                if s not in speedup_ratio:
                    _logger.info(f"cannot get the speed up of strategy {s}")
                    continue

                if speedup_ratio[s] > max_speedup:
                    max_speedup = speedup_ratio[s]
                    final_strategy = s
        else:
            candidate_s = []
            pre_s = None
            for strategy, ratio in sorted_speedup_ratio:
                if abs(ratio - target_speedup) <= 0.1:
                    candidate_s.append(strategy)
                ### if there is no strategy satisfy target speedup
                ### choose the most recent speedup 
                if ratio > target_speedup and len(candidate_s) == 0:
                    if pre_s is not None:
                        candidate_s.append(pre_s)
                    candidate_s.append(strategy)
                pre_s = strategy

            if 'origin_int8' in candidate_s:
                final_strategy = candidate_s
            else:
                candidate_s = sorted(candidate_s, key=lambda x: x.split('_')[1])
                for c in candidate_s:
                    if c.startswith('sparse') and float(c.split('_')[
                            1]) <= MAGIC_SPARSE_RATIO:
                        final_strategy = c

                if final_strategy is None:
                    final_strategy = candidate_s[0]

    ### if deploy_hardware is not None
    else:
        ### default speedup ratio of quantization is 70% compare to fp32
        ### TODO(ceci3): full quant or skip some layer later
        if target_speedup is None:
            if model_type == 'transformer':
                final_strategy = DEFAULT_TRANSFORMER_STRATEGY
            else:
                final_strategy = DEFAULT_STRATEGY

        elif target_speedup > DEFAULT_QUANT_SPEEDUP:
            prune_ratio = target_speedup - DEFAULT_QUANT_SPEEDUP
            if prune_ratio > 1.0:
                raise NotImplementedError(
                    "target_speedup {} is improper".format(target_speedup))
            final_strategy = 'prune_{}_int8'.format(str(prune_ratio))
        else:
            raise NotImplementedError("target_speedup {} is improper".format(
                target_speedup))

    strategy_config = create_strategy_config(final_strategy, model_type)
    return strategy_config


def get_final_quant_config(ptq_loss):
    """ transform quantization tester config to real quantization config """
    if ptq_loss <= MAGIC_EMD_DISTANCE:
        quant_config = Quantization(**default_quant_config)
        hpo_config = HyperParameterOptimization(**default_hpo_config)
        configs = [{
            'Quantization': quant_config,
            'HyperParameterOptimization': hpo_config
        }]

    else:
        quant_config = Quantization(**default_quant_config)
        dis_config = Distillation()
        configs = [{'Quantization': quant_config, 'Distillation': dis_config}]

    return configs


if __name__ == '__main__':
    create_strategy_config('sparse_0.75_int8', 'transformer')
