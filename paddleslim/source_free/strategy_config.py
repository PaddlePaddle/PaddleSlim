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
    "QuantizationConfig", "DistillationConfig", "MultiTeacherDistillationConfig", \
    "HyperParameterOptimizationConfig", "PruneConfig", "UnstructurePruneConfig",  \
    "merge_config", "ProgramInfo", "TrainConfig",
]

### QuantizationConfig:
QuantizationConfig = namedtuple(
    "QuantizationConfig",
    [
        "quantize_op_types",
        "weight_bits",
        "activation_bits",
        "not_quant_pattern",  ### ptq没有暴露相应接口，需要确定是否支持
        "use_pact",  ### 仅QAT支持
        "is_full_quantize"
    ])

QuantizationConfig.__new__.__defaults__ = (None, ) * (
    len(QuantizationConfig._fields) - 1) + (False, )

### DistillationConfig:
DistillationConfig = namedtuple(
    "DistillationConfig",
    [
        "distill_loss",  ### list[list]，支持不同节点之间使用不同的loss。
        "distill_node_pair",  ### list[list]，支持不同节点之间使用不同的loss。
        "distill_lambda",  ### list[list]，支持不同节点之间使用不同的loss。
        "teacher_model_dir",
        "teacher_model_filename",
        "teacher_params_filename",
        "merge_feed",
    ])

DistillationConfig.__new__.__defaults__ = (None, ) * (
    len(DistillationConfig._fields) - 1) + (True, )

### 多teacher蒸馏配置
### Multi-Teacher DistillationConfig：
MultiTeacherDistillationConfig = namedtuple(
    "MultiTeacherDistillationConfig",
    [
        "distill_loss",  ### list[str]，每个teacher对应一个loss
        "distill_node_pair",  ### list[list]，每个teacher对应一个蒸馏。仅支持logits蒸馏，不支持中间层蒸馏
        "distill_lambda",  ### list[float]，每个teacher对应一个lambda。
        "teacher_model_dir",
        "teacher_model_filename",  ### list[str], 每个teacher对应一个模型文件
        "teacher_params_filename",  ### list[str], 每个teacher对应一个参数文件
        "merge_feed",
    ])

MultiTeacherDistillationConfig.__new__.__defaults__ = (None, ) * (
    len(MultiTeacherDistillationConfig._fields) - 1) + (True, )

### 不设置就按照默认的搜索空间进行超参搜索，设置的话按照设置的搜索空间搜索，这样可以支持单PTQ策略
###HyperParameterOptimizationConfig
HyperParameterOptimizationConfig = namedtuple(
    "HyperParameterOptimizationConfig", [
        "ptq_algo", "bias_correct", "weight_quantize_type", "hist_percent",
        "batch_size", "batch_num", "max_quant_count"
    ])

HyperParameterOptimizationConfig.__new__.__defaults__ = (None, ) * (
    len(HyperParameterOptimizationConfig._fields) - 1) + (20, )

### PruneConfig
PruneConfig = namedtuple(
    "PruneConfig",
    [
        "prune_algo",  ### prune, asp
        "pruned_ratio",
        "prune_params_name",
        "criterion",
    ])
PruneConfig.__new__.__defaults__ = (None, ) * len(PruneConfig._fields)

### UnstructurePruneConfig
UnstructurePruneConfig = namedtuple("UnstructurePruneConfig", [
    "prune_strategy",
    "prune_mode",
    "threshold",
    "prune_steps",
    "prune_ratio",
    "initial_ratio",
    "gmp_config",
    "prune_params_type",
    "local_sparsity",
])
UnstructurePruneConfig.__new__.__defaults__ = (
    None, ) * len(UnstructurePruneConfig._fields)

### TrainConfig
TrainConfig = namedtuple("TrainConfig", [
    "epochs",
    "optimizer",
    "optim_args",
    "learning_rate",
    "lr_decay",
    "eval_iter",
    "logging_iter",
    "origin_metric",
    "target_metric",
])

TrainConfig.__new__.__defaults__ = (None, ) * len(TrainConfig._fields)


def merge_config(*args):
    fields = tuple()
    cfg = dict()
    for arg in args:
        fields += arg._fields
        cfg.update(dict(arg._asdict()))
    MergeConfig = namedtuple("MergeConfig", fields)
    return MergeConfig(**cfg)


#ProgramInfo = namedtuple("ProgramInfo", [
#    "startup_program", "program", "feed_target_names", "fetch_targets",
#    "optimizer"
#])
#ProgramInfo.__new__.__defaults__ = (None, ) * len(ProgramInfo._fields)
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
