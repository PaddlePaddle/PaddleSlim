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
import paddle.nn as nn
from paddle.nn import ReLU
from paddleslim.nas import ofa
from paddleslim.nas.ofa import OFA, RunConfig, DistillConfig
from paddleslim.nas.ofa.convert_super import supernet
from paddleslim.nas.ofa.layers import Block, SuperSeparableConv2D


class ModelConv(nn.Layer):
    def __init__(self):
        super(ModelConv, self).__init__()
        with supernet(
                kernel_size=(3, 5, 7),
                channel=((4, 8, 12), (8, 12, 16), (8, 12, 16),
                         (8, 12, 16))) as ofa_super:
            models = []
            models += [nn.Conv2D(3, 4, 3, padding=1)]
            models += [nn.InstanceNorm2D(4)]
            models += [ReLU()]
            models += [nn.Conv2D(4, 4, 3, groups=4)]
            models += [nn.InstanceNorm2D(4)]
            models += [ReLU()]
            models += [nn.Conv2DTranspose(4, 4, 3, groups=4, padding=1)]
            models += [nn.BatchNorm2D(4)]
            models += [ReLU()]
            models += [nn.Conv2D(4, 3, 3)]
            models += [ReLU()]
            models = ofa_super.convert(models)

        models += [
            Block(
                SuperSeparableConv2D(
                    3, 6, 1, padding=1, candidate_config={'channel': (3, 6)}),
                fixed=True)
        ]
        with supernet(
                kernel_size=(3, 5, 7), expand_ratio=(1, 2, 4)) as ofa_super:
            models1 = []
            models1 += [nn.Conv2D(6, 4, 3)]
            models1 += [nn.BatchNorm2D(4)]
            models1 += [ReLU()]
            models1 += [nn.Conv2D(4, 4, 3, groups=2)]
            models1 += [nn.InstanceNorm2D(4)]
            models1 += [ReLU()]
            models1 += [nn.Conv2DTranspose(4, 4, 3, groups=2)]
            models1 += [nn.BatchNorm2D(4)]
            models1 += [ReLU()]
            models1 += [nn.Conv2DTranspose(4, 4, 3)]
            models1 += [nn.BatchNorm2D(4)]
            models1 += [ReLU()]
            models1 += [nn.Conv2DTranspose(4, 4, 1)]
            models1 += [nn.BatchNorm2D(4)]
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


class ModelConv2(nn.Layer):
    def __init__(self):
        super(ModelConv2, self).__init__()
        with supernet(expand_ratio=(1, 2, 4)) as ofa_super:
            models = []
            models += [
                nn.Conv2DTranspose(
                    4, 4, 3, weight_attr=paddle.ParamAttr(name='conv1_w'))
            ]
            models += [
                nn.BatchNorm2D(
                    4,
                    weight_attr=paddle.ParamAttr(name='bn1_w'),
                    bias_attr=paddle.ParamAttr(name='bn1_b'))
            ]
            models += [ReLU()]
            models += [nn.Conv2D(4, 4, 3)]
            models += [nn.BatchNorm2D(4)]
            models += [ReLU()]
            models = ofa_super.convert(models)

        with supernet(channel=((4, 6, 8), (4, 6, 8))) as ofa_super:
            models1 = []
            models1 += [nn.Conv2DTranspose(4, 4, 3)]
            models1 += [nn.BatchNorm2D(4)]
            models1 += [ReLU()]
            models1 += [nn.Conv2DTranspose(4, 4, 3)]
            models1 += [nn.BatchNorm2D(4)]
            models1 += [ReLU()]
            models1 = ofa_super.convert(models1)
        models += models1

        with supernet(kernel_size=(3, 5, 7)) as ofa_super:
            models2 = []
            models2 += [nn.Conv2D(4, 4, 3)]
            models2 += [nn.BatchNorm2D(4)]
            models2 += [ReLU()]
            models2 += [nn.Conv2DTranspose(4, 4, 3)]
            models2 += [nn.BatchNorm2D(4)]
            models2 += [ReLU()]
            models2 += [nn.Conv2D(4, 4, 3)]
            models2 += [nn.BatchNorm2D(4)]
            models2 += [ReLU()]
            models2 = ofa_super.convert(models2)

        models += models2
        self.models = paddle.nn.Sequential(*models)


class ModelLinear(nn.Layer):
    def __init__(self):
        super(ModelLinear, self).__init__()
        with supernet(expand_ratio=(1.0, 2.0, 4.0)) as ofa_super:
            models = []
            models += [nn.Embedding(num_embeddings=64, embedding_dim=64)]
            models += [nn.Linear(64, 128)]
            models += [nn.LayerNorm(128)]
            models += [nn.Linear(128, 256)]
            models = ofa_super.convert(models)

        with supernet(expand_ratio=(1, 2, 4)) as ofa_super:
            models1 = []
            models1 += [nn.Linear(256, 256)]
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


class ModelOriginLinear(nn.Layer):
    def __init__(self):
        super(ModelOriginLinear, self).__init__()
        models = []
        models += [nn.Embedding(num_embeddings=64, embedding_dim=64)]
        models += [nn.Linear(64, 128)]
        models += [nn.LayerNorm(128)]
        models += [nn.Linear(128, 256)]
        models += [nn.Linear(256, 256)]

        self.models = paddle.nn.Sequential(*models)

    def forward(self, inputs):
        return self.models(inputs)


class ModelLinear1(nn.Layer):
    def __init__(self):
        super(ModelLinear1, self).__init__()
        with supernet(channel=((64, 128, 256), (64, 128, 256),
                               (64, 128, 256))) as ofa_super:
            models = []
            models += [nn.Embedding(num_embeddings=64, embedding_dim=64)]
            models += [nn.Linear(64, 128)]
            models += [nn.LayerNorm(128)]
            models += [nn.Linear(128, 256)]
            models = ofa_super.convert(models)

        with supernet(channel=((64, 128, 256), )) as ofa_super:
            models1 = []
            models1 += [nn.Linear(256, 256)]
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


class ModelLinear2(nn.Layer):
    def __init__(self):
        super(ModelLinear2, self).__init__()
        with supernet(expand_ratio=None) as ofa_super:
            models = []
            models += [
                nn.Embedding(
                    num_embeddings=64,
                    embedding_dim=64,
                    weight_attr=paddle.ParamAttr(name='emb'))
            ]
            models += [
                nn.Linear(
                    64,
                    128,
                    weight_attr=paddle.ParamAttr(name='fc1_w'),
                    bias_attr=paddle.ParamAttr(name='fc1_b'))
            ]
            models += [
                nn.LayerNorm(
                    128,
                    weight_attr=paddle.ParamAttr(name='ln1_w'),
                    bias_attr=paddle.ParamAttr(name='ln1_b'))
            ]
            models += [nn.Linear(128, 256)]
            models = ofa_super.convert(models)
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


class TestOFA(unittest.TestCase):
    def setUp(self):
        self.init_model_and_data()
        self.init_config()

    def init_model_and_data(self):
        self.model = ModelConv()
        self.teacher_model = ModelConv()
        data_np = np.random.random((1, 3, 10, 10)).astype(np.float32)
        label_np = np.random.random((1)).astype(np.float32)

        self.data = paddle.to_tensor(data_np)

    def init_config(self):
        default_run_config = {
            'train_batch_size': 1,
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
            'mapping_layers': ['models.0.fn'],
            'mapping_op': 'conv2d'
        }
        self.distill_config = DistillConfig(**default_distill_config)
        self.elastic_order = ['kernel_size', 'width', 'depth']

    def test_ofa(self):
        ofa_model = OFA(self.model,
                        self.run_config,
                        distill_config=self.distill_config,
                        elastic_order=self.elastic_order)

        start_epoch = 0
        for idx in range(len(self.run_config.n_epochs)):
            cur_idx = self.run_config.n_epochs[idx]
            for ph_idx in range(len(cur_idx)):
                cur_lr = self.run_config.init_learning_rate[idx][ph_idx]
                adam = paddle.optimizer.Adam(
                    learning_rate=cur_lr,
                    parameters=(ofa_model.parameters() + ofa_model.netAs_param))
                for epoch_id in range(start_epoch,
                                      self.run_config.n_epochs[idx][ph_idx]):
                    if epoch_id == 0:
                        ofa_model.set_epoch(epoch_id)
                    for model_no in range(self.run_config.dynamic_batch_size[
                            idx]):
                        output, _ = ofa_model(self.data)
                        loss = paddle.mean(output)
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
        self.data = paddle.to_tensor(data_np)

    def init_config(self):
        default_run_config = {
            'train_batch_size': 1,
            'n_epochs': [[2, 5]],
            'init_learning_rate': [[0.003, 0.001]],
            'dynamic_batch_size': [1],
            'total_images': 1,
        }
        self.run_config = RunConfig(**default_run_config)

        default_distill_config = {
            'lambda_distill': 0.01,
            'teacher_model': self.teacher_model,
            'mapping_op': 'linear',
            'mapping_layers': ['models.3.fn'],
        }
        self.distill_config = DistillConfig(**default_distill_config)
        self.elastic_order = None


class TestOFACase2(TestOFA):
    def init_model_and_data(self):
        self.model = ModelLinear1()
        self.teacher_model = ModelLinear1()
        data_np = np.random.random((3, 64)).astype(np.int64)

        self.data = paddle.to_tensor(data_np)

    def init_config(self):
        default_run_config = {
            'train_batch_size': 1,
            'n_epochs': [[2, 5]],
            'init_learning_rate': [[0.003, 0.001]],
            'dynamic_batch_size': [1],
            'total_images': 1,
        }
        self.run_config = RunConfig(**default_run_config)
        default_distill_config = {
            'teacher_model': self.teacher_model,
            'mapping_layers': ['models.3.fn'],
        }
        self.distill_config = DistillConfig(**default_distill_config)
        self.elastic_order = None


class TestOFACase3(unittest.TestCase):
    def test_ofa(self):
        self.model = ModelLinear2()
        ofa_model = OFA(self.model)
        ofa_model.set_net_config({'expand_ratio': None})


class TestOFACase4(unittest.TestCase):
    def test_ofa(self):
        self.model = ModelConv2()


class TestExport(unittest.TestCase):
    def setUp(self):
        self._init_model()

    def _init_model(self):
        self.origin_model = ModelOriginLinear()
        model = ModelLinear()
        self.ofa_model = OFA(model)

    def test_ofa(self):
        config = {
            'embedding_1': {
                'expand_ratio': (2.0)
            },
            'linear_3': {
                'expand_ratio': (2.0)
            },
            'linear_4': {},
            'linear_5': {}
        }
        origin_dict = {}
        for name, param in self.origin_model.named_parameters():
            origin_dict[name] = param.shape
        self.ofa_model.export(
            self.origin_model,
            config,
            input_shapes=[[1, 64]],
            input_dtypes=['int64'])
        for name, param in self.origin_model.named_parameters():
            if name in config.keys():
                if 'expand_ratio' in config[name]:
                    assert origin_dict[name][-1] == param.shape[-1] * config[
                        name]['expand_ratio']


if __name__ == '__main__':
    unittest.main()
