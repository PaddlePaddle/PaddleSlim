# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
from .flops import flops, dygraph_flops
from .model_size import model_size
from .latency import LatencyEvaluator, TableLatencyEvaluator
from .latency_predictor import LatencyPredictor, TableLatencyPredictor
from .parse_ops import get_key_from_op
from ._utils import save_cls_model, save_det_model

__all__ = [
    'flops', 'dygraph_flops', 'model_size', 'LatencyEvaluator',
    'TableLatencyEvaluator', "LatencyPredictor", "TableLatencyPredictor",
    "get_key_from_op", "save_cls_model", "save_det_model"
]
