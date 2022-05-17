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

from collections import namedtuple

__all__ = [
    "Quantization", "Distillation", "MultiTeacherDistillation", \
    "HyperParameterOptimization", "Prune", "UnstructurePrune",  \
    "merge_config", "ProgramInfo", "TrainConfig",
]

### Quantization:
Quantization = namedtuple(
    "Quantization",
    [
        "quantize_op_types",
        "weight_bits",
        "activation_bits",
        "not_quant_pattern",  # Only support in QAT
        "use_pact",  # Only support in QAT
        "is_full_quantize",
        "activation_quantize_type",
        "weight_quantize_type"
    ])

Quantization.__new__.__defaults__ = (None, ) * (len(Quantization._fields) - 1
                                                ) + (False, )

### Distillation:
Distillation = namedtuple(
    "Distillation",
    [
        "distill_loss",  ### list[list]，支持不同节点之间使用不同的loss。
        "distill_node_pair",  ### list[list]，支持不同节点之间使用不同的loss。
        "distill_lambda",  ### list[list]，支持不同节点之间使用不同的loss。
        "teacher_model_dir",
        "teacher_model_filename",
        "teacher_params_filename",
        "merge_feed",
    ])

Distillation.__new__.__defaults__ = (None, ) * (len(Distillation._fields) - 1
                                                ) + (True, )

### 多teacher蒸馏配置
### Multi-Teacher Distillation：
MultiTeacherDistillation = namedtuple(
    "MultiTeacherDistillation",
    [
        "distill_loss",  ### list[str]，每个teacher对应一个loss
        "distill_node_pair",  ### list[list]，每个teacher对应一个蒸馏。仅支持logits蒸馏，不支持中间层蒸馏
        "distill_lambda",  ### list[float]，每个teacher对应一个lambda。
        "teacher_model_dir",
        "teacher_model_filename",  ### list[str], 每个teacher对应一个模型文件
        "teacher_params_filename",  ### list[str], 每个teacher对应一个参数文件
        "merge_feed",
    ])

MultiTeacherDistillation.__new__.__defaults__ = (None, ) * (
    len(MultiTeacherDistillation._fields) - 1) + (True, )

### 不设置就按照默认的搜索空间进行超参搜索，设置的话按照设置的搜索空间搜索，这样可以支持单PTQ策略
###HyperParameterOptimization
HyperParameterOptimization = namedtuple("HyperParameterOptimization", [
    "ptq_algo", "bias_correct", "weight_quantize_type", "hist_percent",
    "batch_num", "max_quant_count"
])

HyperParameterOptimization.__new__.__defaults__ = (None, ) * (
    len(HyperParameterOptimization._fields) - 1) + (20, )

### Prune
Prune = namedtuple("Prune", [
    "prune_algo",
    "pruned_ratio",
    "prune_params_name",
    "criterion",
])
Prune.__new__.__defaults__ = (None, ) * len(Prune._fields)

### UnstructurePrune
UnstructurePrune = namedtuple("UnstructurePrune", [
    "prune_strategy",
    "prune_mode",
    "threshold",
    "pruned_ratio",
    "gmp_config",
    "prune_params_type",
    "local_sparsity",
])
UnstructurePrune.__new__.__defaults__ = (None, ) * len(UnstructurePrune._fields)

### Train
TrainConfig = namedtuple("Train", [
    "epochs", "learning_rate", "optimizer", "optim_args", "eval_iter",
    "logging_iter", "origin_metric", "target_metric", "use_fleet", "amp_config",
    "recompute_config", "sharding_config", "sparse_model"
])

TrainConfig.__new__.__defaults__ = (None, ) * len(TrainConfig._fields)


def merge_config(*args):
    fields = set()
    cfg = dict()
    for arg in args:
        fields = fields.union(arg._fields)
        cfg.update(dict(arg._asdict()))
    MergeConfig = namedtuple("MergeConfig", fields)
    return MergeConfig(**cfg)


class ProgramInfo:
    def __init__(self,
                 startup_program,
                 program,
                 feed_target_names,
                 fetch_targets,
                 optimizer=None):
        self.startup_program = startup_program
        self.program = program
        self.feed_target_names = feed_target_names
        self.fetch_targets = fetch_targets
        self.optimizer = optimizer
