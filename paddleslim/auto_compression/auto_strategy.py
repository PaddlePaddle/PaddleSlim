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
import platform
from .utils.predict import predict_compressed_model
from .strategy_config import *

__all__ = [
    "auto_prepare_strategy", "create_strategy_config", "get_final_quant_config"
]

hpo_config_tester = {
    "ptq_algo": ["avg"],
    "weight_quantize_type": ['channel_wise_abs_max', 'abs_max'],
    "bias_correct": [False],
    "batch_num": [2, 3],
    "max_quant_count": 1,
}

default_hpo_config = {
    "ptq_algo": ["KL", "hist", "avg", "mse"],
    "weight_quantize_type": ['channel_wise_abs_max', 'abs_max'],
    "bias_correct": [True, False],
    "hist_percent": [0.98, 0.999],
    "batch_num": [10, 30],
    "max_quant_count": 20,
}

default_quant_config = {
    'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul', 'matmul'],
    'weight_bits': 8,
    'activation_bits': 8
}


def create_strategy_config(strategy_str, model_type):
    tmp_s = strategy_str.split('_')
    configs = []
    ### only quant
    dis_config = Distillation()
    if len(tmp_s) == 3:
        tmp_s[0] = tmp_s[0].replace('prune', 'Prune')
        tmp_s[0] = tmp_s[0].replace('sparse', 'UnstructurePrune')
        ### TODO(ceci3): auto chooice prune algo
        default_prune_config = {
            'prune_ratio': float(tmp_s[1]),
            'prune_algo': 'prune',
            'criterion': 'l1_norm'
        }
        if model_type == 'transformer' and tmp_s[0] == 'Prune':
            default_prune_config['prune_algo'] = 'transformer_pruner'
        prune_config = eval(tmp_s[0])(**default_prune_config)
        configs.append({tmp_s[0]: prune_config, 'Distillation': dis_config})

    if tmp_s[-1] == 'int8':
        ### TODO(ceci3): support skip some layer and full quant
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
            configs = [{
                'Quantization': quant_config,
                'Distillation': dis_config
            }]

    return configs


def auto_prepare_strategy(model_dir,
                          model_filename,
                          params_filename,
                          target_speedup=None,
                          deploy_hardware=None,
                          model_type=None):
    final_strategy = None

    model_file = os.path.join(model_dir, model_filename)
    param_file = os.path.join(model_dir, params_filename)

    if deploy_hardware is not None:
        compressed_time_dict = predict_compressed_model(
            model_file, param_file, hardware=deploy_hardware)

        baseline = compressed_time_dict['origin_fp32']
        speedup_ratio = {}
        for strategy, latency in compressed_time_dict.items():
            speedup_ratio[strategy] = 1.0 - float(latency) / baseline

        sorted_speedup_ratio = sorted(
            speedup_ratio.items(), key=lambda x: x[1])  #, reverse=True)

        ### if target speedup is None, choose strategy by experience.
        if target_speedup is None:
            max_speedup_strategy = -1.0
            for s in [
                    'sparse_0.75_fp32', 'prune_0.3_fp32', 'origin_int8',
                    'sparse_0.75_int8', 'prune_0.3_int8'
            ]:
                if speedup_ratio[s] > max_speedup_strategy:
                    max_speedup_strategy = speedup_ratio[s]
                    final_strategy = s
        else:
            candidate_s = []
            pre_s = None
            for strategy, ratio in sorted_speedup_ratio:
                if abs(ratio - target_speedup) <= 0.1:
                    candidate_s.append(strategy)
                ### if there is no strategy satisfy target speedup
                if ratio > target_speedup and len(candidate_s) == 0:
                    candidate_s.append(pre_s)
                    candidate_s.append(strategy)
                pre_s = strategy

            if 'origin_int8' in candidate_s:
                final_strategy = candidate_s
            else:
                candidate_ratio = sorted(
                    candidate_s, key=lambda x: x.split('_')[1])
                for c in candidate_ratio:
                    if c.startswith('sparse') and float(c.split('_')[
                            1]) <= 0.75:
                        final_strategy = c

                if final_strategy is None:
                    final_strategy = candidate_ratio[0]

    else:
        ### default quant speedup ratio is 70% compare to fp32
        ### TODO(ceci3): full quant or skip some layer later
        if target_speedup is None:
            if model_type == 'transformer':
                final_strategy = 'prune_0.25_int8'
            else:
                final_strategy = 'origin_int8'

        elif target_speedup > 0.7:
            prune_ratio = target_speedup - 0.7
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
    ### TODO: 0.03 threshold maybe not suitable, need to check
    if ptq_loss <= 0.03:
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
