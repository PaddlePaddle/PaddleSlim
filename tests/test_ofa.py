# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys
sys.path.append("../")
import numpy as np
import unittest
import paddle
from static_case import StaticCase
import paddle.fluid as fluid
import paddle.nn as nn
from paddle.nn import ReLU
from paddleslim.nas import ofa
from paddleslim.nas.ofa import OFA, RunConfig, DistillConfig
from paddleslim.nas.ofa.convert_super import supernet
from paddleslim.nas.ofa.layers import Block, SuperSeparableConv2D


class ModelConv(fluid.dygraph.Layer):
    def __init__(self):
        super(ModelConv, self).__init__()
        with supernet(
                kernel_size=(3, 5, 7),
                channel=((4, 8, 12), (8, 12, 16), (8, 12, 16),
                         (8, 12, 16))) as ofa_super:
            models = []
            models += [nn.Conv2D(3, 4, 3, padding=1)]
            models += [nn.InstanceNorm(4)]
            models += [ReLU()]
            models += [nn.Conv2D(4, 4, 3, groups=4)]
            models += [nn.InstanceNorm(4)]
            models += [ReLU()]
            models += [
                nn.Conv2DTranspose(
                    4, 4, 3, groups=4, padding=1, use_cudnn=True)
            ]
            models += [nn.BatchNorm(4)]
            models += [ReLU()]
            models += [nn.Conv2D(4, 3, 3)]
            models += [ReLU()]
            models = ofa_super.convert(models)

        models += [
            Block(
                SuperSeparableConv2D(
                    3, 6, 1, padding=1, candidate_config={'channel': (3, 6)}))
        ]
        with supernet(
                kernel_size=(3, 5, 7), expand_ratio=(1, 2, 4)) as ofa_super:
            models1 = []
            models1 += [nn.Conv2D(6, 4, 3)]
            models1 += [nn.BatchNorm(4)]
            models1 += [ReLU()]
            models1 += [nn.Conv2D(4, 4, 3, groups=2)]
            models1 += [nn.InstanceNorm(4)]
            models1 += [ReLU()]
            models1 += [nn.Conv2DTranspose(4, 4, 3, groups=2)]
            models1 += [nn.BatchNorm(4)]
            models1 += [ReLU()]
            models1 += [nn.Conv2DTranspose(4, 4, 3)]
            models1 += [nn.BatchNorm(4)]
            models1 += [ReLU()]
            models1 = ofa_super.convert(models1)

        models += models1

        self.models = paddle.nn.Sequential(*models)

    def forward(self, inputs, depth=None):
        if depth != None:
            assert isinstance(depth, int)
            assert depth <= len(self.models)
        else:
            depth = len(self.models)
        for idx in range(depth):
            layer = self.models[idx]
            inputs = layer(inputs)
        return inputs


class ModelLinear(fluid.dygraph.Layer):
    def __init__(self):
        super(ModelLinear, self).__init__()
        models = []
        with supernet(expand_ratio=(1, 2, 4)) as ofa_super:
            models1 = []
            models1 += [nn.Embedding(size=(64, 64))]
            models1 += [nn.Linear(64, 128)]
            models1 += [nn.LayerNorm(128)]
            models1 += [nn.Linear(128, 256)]
            models1 = ofa_super.convert(models1)

        models += models1

        with supernet(channel=((64, 128, 256), (64, 128, 256))) as ofa_super:
            models1 = []
            models1 += [nn.Linear(256, 128)]
            models1 += [nn.LayerNorm(128)]
            models1 += [nn.Linear(128, 256)]
            models1 = ofa_super.convert(models1)

        models += models1

        self.models = paddle.nn.Sequential(*models)

    def forward(self, inputs, depth=None):
        if depth != None:
            assert isinstance(depth, int)
            assert depth < len(self.models)
        else:
            depth = len(self.models)
        for idx in range(depth):
            layer = self.models[idx]
            inputs = layer(inputs)
        return inputs


class TestOFA(StaticCase):
    def setUp(self):
        fluid.enable_dygraph()
        self.init_model_and_data()
        self.init_config()

    def init_model_and_data(self):
        self.model = ModelConv()
        self.teacher_model = ModelConv()
        data_np = np.random.random((1, 3, 10, 10)).astype(np.float32)
        label_np = np.random.random((1)).astype(np.float32)

        self.data = fluid.dygraph.to_variable(data_np)

    def init_config(self):
        default_run_config = {
            'train_batch_size': 1,
            'eval_batch_size': 1,
            'n_epochs': [[1], [2, 3], [4, 5]],
            'init_learning_rate': [[0.001], [0.003, 0.001], [0.003, 0.001]],
            'dynamic_batch_size': [1, 1, 1],
            'total_images': 1,
            'elastic_depth': (5, 15, 24)
        }
        self.run_config = RunConfig(**default_run_config)

        default_distill_config = {
            'lambda_distill': 0.01,
            'teacher_model': self.teacher_model,
            'mapping_layers': ['models.0.fn']
        }
        self.distill_config = DistillConfig(**default_distill_config)

    def test_ofa(self):
        ofa_model = OFA(self.model,
                        self.run_config,
                        distill_config=self.distill_config)

        start_epoch = 0
        for idx in range(len(self.run_config.n_epochs)):
            cur_idx = self.run_config.n_epochs[idx]
            for ph_idx in range(len(cur_idx)):
                cur_lr = self.run_config.init_learning_rate[idx][ph_idx]
                adam = fluid.optimizer.Adam(
                    learning_rate=cur_lr,
                    parameter_list=(
                        ofa_model.parameters() + ofa_model.netAs_param))
                for epoch_id in range(start_epoch,
                                      self.run_config.n_epochs[idx][ph_idx]):
                    if epoch_id == 0:
                        ofa_model.set_epoch(epoch_id)
                    for model_no in range(self.run_config.dynamic_batch_size[
                            idx]):
                        output, _ = ofa_model(self.data)
                        if model_no == 0:
                            first_net_config = ofa_model.current_config
                            ofa_model.set_net_config(first_net_config)
                        loss = fluid.layers.reduce_mean(output)
                        if self.distill_config.mapping_layers != None:
                            dis_loss = ofa_model.calc_distill_loss()
                            loss += dis_loss
                            dis_loss = dis_loss.numpy()[0]
                        else:
                            dis_loss = 0
                        print('epoch: {}, loss: {}, distill loss: {}'.format(
                            epoch_id, loss.numpy()[0], dis_loss))
                        loss.backward()
                        adam.minimize(loss)
                        adam.clear_gradients()
                start_epoch = self.run_config.n_epochs[idx][ph_idx]


class TestOFACase1(TestOFA):
    def init_model_and_data(self):
        self.model = ModelLinear()
        self.teacher_model = ModelLinear()
        data_np = np.random.random((3, 64)).astype(np.int64)

        self.data = fluid.dygraph.to_variable(data_np)

    def init_config(self):
        default_run_config = {
            'train_batch_size': 1,
            'eval_batch_size': 1,
            'n_epochs': [[2, 5]],
            'init_learning_rate': [[0.003, 0.001]],
            'dynamic_batch_size': [1],
            'total_images': 1,
        }
        self.run_config = RunConfig(**default_run_config)

        default_distill_config = {
            'lambda_distill': 0.01,
            'teacher_model': self.teacher_model,
        }
        self.distill_config = DistillConfig(**default_distill_config)


if __name__ == '__main__':
    unittest.main()
