# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import copy
from paddle.quantization import QAT, QuantConfig
from ..core import GraphTracer


class SlimQAT(QAT):
    def __init__(self, q_config):
        super(SlimQAT, self).__init__(q_config)

    def _apply_constraints(self, model, graph):
        for _contraint in self._config.constraints:
            _contraint.apply(model, graph, self._config)

    def _analysis_model(self, _model, inputs):
        assert inputs is not None
        tracer = GraphTracer(_model)
        tracer(inputs)
        return tracer.graph

    def quantize(self, model: paddle.nn.Layer, inplace=False, inputs=None):
        _model = model if inplace else copy.deepcopy(model)
        self._config._specify(_model)
        graph = self._analysis_model(_model, inputs)
        self._apply_constraints(_model, graph)
        self._convert_to_quant_layers(_model, self._config)
        self._insert_activation_observers(_model, self._config)
        return _model
