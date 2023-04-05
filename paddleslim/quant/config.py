# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.quantization.config import QuantConfig
from paddle.quantization.factory import QuanterFactory


class SlimQuantConfig(QuantConfig):
    def __init__(self, activation: QuanterFactory, weight: QuanterFactory):
        super(SlimQuantConfig, self).__init__(activation, weight)
        self._constraints = []

    @property
    def constraints(self):
        return self._constraints

    def add_constraints(self, constraints):
        if not isinstance(constraints, (list, tuple)):
            constraints = [constraints]
        self._constraints.extend(constraints)