#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.nn as nn
import paddle.fluid as fluid


class AccuracyEvaluator:
    def __init__(self, model=None, input_dim=128):
        if model == None:
            self.model = DefaultModel(input_dim=input_dim)
        else:
            assert isinstance(model, fluid.dygraph.Layer)
            self.model = model

    @fluid.dygraph.no_grad
    def predict_accuracy(self, net_arch):
        pred = self.model(net_arch)
        return pred.numpy()

    def convert_net_to_onehot(self, net):
        pass

    def convert_onehot_to_net(self, net_onehot):
        pass


class DefaultModel(fluid.dygraph.Layer):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.models = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(), nn.Linear(400, 400), nn.ReLU(), nn.Linear(400, 1))

    def forward(self, *inputs, **kwargs):
        return self.model(inputs)
