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
from paddleslim.nas import ofa
from paddleslim.nas.ofa import OFA, RunConfig, DistillConfig
from paddleslim.nas.ofa.convert_super import supernet
from paddleslim.nas.ofa.layers import Block, SuperSeparableConv2D
from paddleslim.nas.ofa.convert_super import Convert, supernet


class ModelConv(paddle.nn.Layer):
    def __init__(self):
        super(ModelConv, self).__init__()
        with supernet(
                kernel_size=(3, 4, 5, 7),
                channel=((4, 8, 12), (8, 12, 16), (8, 12, 16),
                         (8, 12, 16))) as ofa_super:
            models = []
            models += [paddle.nn.Conv2D(3, 4, 3, padding=1)]
            models += [paddle.nn.InstanceNorm2D(4)]
            models += [paddle.nn.SyncBatchNorm(4)]
            models += [paddle.nn.ReLU()]
            models += [paddle.nn.Conv2D(4, 4, 3, groups=4)]
            models += [paddle.nn.InstanceNorm2D(4)]
            models += [paddle.nn.ReLU()]
            models += [paddle.nn.Conv2DTranspose(4, 4, 3, groups=4, padding=1)]
            models += [paddle.nn.BatchNorm2D(4)]
            models += [paddle.nn.ReLU()]
            models += [paddle.nn.Conv2D(4, 3, 3)]
            models += [paddle.nn.ReLU()]
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
            models1 += [paddle.nn.Conv2D(6, 4, 3)]
            models1 += [paddle.nn.BatchNorm2D(4)]
            models1 += [paddle.nn.ReLU()]
            models1 += [paddle.nn.Conv2D(4, 4, 3, groups=2)]
            models1 += [paddle.nn.InstanceNorm2D(4)]
            models1 += [paddle.nn.ReLU()]
            models1 += [paddle.nn.Conv2DTranspose(4, 4, 3, groups=2)]
            models1 += [paddle.nn.BatchNorm2D(4)]
            models1 += [paddle.nn.ReLU()]
            models1 += [paddle.nn.Conv2DTranspose(4, 4, 3)]
            models1 += [paddle.nn.BatchNorm2D(4)]
            models1 += [paddle.nn.ReLU()]
            models1 += [paddle.nn.Conv2DTranspose(4, 4, 1)]
            models1 += [paddle.nn.BatchNorm2D(4)]
            models1 += [paddle.nn.ReLU()]
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


class ModelConv2(paddle.nn.Layer):
    def __init__(self):
        super(ModelConv2, self).__init__()
        with supernet(expand_ratio=(1, 2, 4)) as ofa_super:
            models = []
            models += [
                paddle.nn.Conv2DTranspose(
                    4, 4, 3, weight_attr=paddle.ParamAttr(name='conv1_w'))
            ]
            models += [
                paddle.nn.BatchNorm2D(
                    4,
                    weight_attr=paddle.ParamAttr(name='bn1_w'),
                    bias_attr=paddle.ParamAttr(name='bn1_b'))
            ]
            models += [paddle.nn.ReLU()]
            models += [paddle.nn.Conv2D(4, 4, 3)]
            models += [paddle.nn.BatchNorm2D(4)]
            models += [paddle.nn.ReLU()]
            models = ofa_super.convert(models)

        with supernet(channel=((4, 6, 8), (4, 6, 8))) as ofa_super:
            models1 = []
            models1 += [paddle.nn.Conv2DTranspose(4, 4, 3)]
            models1 += [paddle.nn.BatchNorm2D(4)]
            models1 += [paddle.nn.ReLU()]
            models1 += [paddle.nn.Conv2DTranspose(4, 4, 3)]
            models1 += [paddle.nn.BatchNorm2D(4)]
            models1 += [paddle.nn.ReLU()]
            models1 = ofa_super.convert(models1)
        models += models1

        with supernet(kernel_size=(3, 5, 7)) as ofa_super:
            models2 = []
            models2 += [paddle.nn.Conv2D(4, 4, 3)]
            models2 += [paddle.nn.BatchNorm2D(4)]
            models2 += [paddle.nn.ReLU()]
            models2 += [paddle.nn.Conv2DTranspose(4, 4, 3)]
            models2 += [paddle.nn.BatchNorm2D(4)]
            models2 += [paddle.nn.ReLU()]
            models2 += [paddle.nn.Conv2D(4, 4, 3)]
            models2 += [paddle.nn.BatchNorm2D(4)]
            models2 += [paddle.nn.ReLU()]
            models2 = ofa_super.convert(models2)

        models += models2
        self.models = paddle.nn.Sequential(*models)


class ModelLinear(paddle.nn.Layer):
    def __init__(self):
        super(ModelLinear, self).__init__()
        with supernet(expand_ratio=(1, 2, 4)) as ofa_super:
            models = []
            models += [paddle.nn.Embedding(num_embeddings=64, embedding_dim=64)]
            weight_attr = paddle.ParamAttr(
                learning_rate=0.5,
                regularizer=paddle.regularizer.L2Decay(1.0),
                trainable=True)
            bias_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0))
            models += [
                paddle.nn.Linear(
                    64, 128, weight_attr=weight_attr, bias_attr=bias_attr)
            ]
            models += [paddle.nn.LayerNorm(128)]
            models += [paddle.nn.Linear(128, 256)]
            models = ofa_super.convert(models)

        with supernet(expand_ratio=(1, 2, 4)) as ofa_super:
            models1 = []
            models1 += [paddle.nn.Linear(256, 256)]
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


class ModelOriginLinear(paddle.nn.Layer):
    def __init__(self):
        super(ModelOriginLinear, self).__init__()
        models = []
        models += [paddle.nn.Embedding(num_embeddings=64, embedding_dim=64)]
        models += [paddle.nn.Linear(64, 128)]
        models += [paddle.nn.LayerNorm(128)]
        models += [paddle.nn.Linear(128, 256)]
        models += [paddle.nn.Linear(256, 256)]

        self.models = paddle.nn.Sequential(*models)

    def forward(self, inputs):
        return self.models(inputs)


class ModelLinear1(paddle.nn.Layer):
    def __init__(self):
        super(ModelLinear1, self).__init__()
        with supernet(channel=((64, 128, 256), (64, 128, 256),
                               (64, 128, 256))) as ofa_super:
            models = []
            models += [paddle.nn.Embedding(num_embeddings=64, embedding_dim=64)]
            models += [paddle.nn.Linear(64, 128)]
            models += [paddle.nn.LayerNorm(128)]
            models += [paddle.nn.Linear(128, 256)]
            models = ofa_super.convert(models)

        with supernet(channel=((64, 128, 256), )) as ofa_super:
            models1 = []
            models1 += [paddle.nn.Linear(256, 256)]
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


class ModelLinear2(paddle.nn.Layer):
    def __init__(self):
        super(ModelLinear2, self).__init__()
        with supernet(expand_ratio=None) as ofa_super:
            models = []
            models += [
                paddle.nn.Embedding(
                    num_embeddings=64,
                    embedding_dim=64,
                    weight_attr=paddle.ParamAttr(name='emb'))
            ]
            models += [
                paddle.nn.Linear(
                    64,
                    128,
                    weight_attr=paddle.ParamAttr(name='fc1_w'),
                    bias_attr=paddle.ParamAttr(name='fc1_b'))
            ]
            models += [
                paddle.nn.LayerNorm(
                    128,
                    weight_attr=paddle.ParamAttr(name='ln1_w'),
                    bias_attr=paddle.ParamAttr(name='ln1_b'))
            ]
            models += [paddle.nn.Linear(128, 256)]
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
                        output = ofa_model(self.data)
                        if (isinstance(output, tuple)):
                            output = output[0]
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
        teacher_model_state_dict = self.teacher_model.state_dict()
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
            'teacher_model_path': teacher_model_state_dict
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
        config = self.ofa_model._sample_config(task='expand_ratio', phase=None)
        origin_dict = {}
        for name, param in self.origin_model.named_parameters():
            origin_dict[name] = param.shape
        self.ofa_model.export(
            config,
            input_shapes=[[1, 64]],
            input_dtypes=['int64'],
            origin_model=self.origin_model)
        for name, param in self.origin_model.named_parameters():
            if name in config.keys():
                if 'expand_ratio' in config[name]:
                    assert origin_dict[name][-1] == param.shape[-1] * config[
                        name]['expand_ratio']


class TestShortCut(unittest.TestCase):
    def setUp(self):
        model = paddle.vision.models.resnet50()
        sp_net_config = supernet(expand_ratio=[0.25, 0.5, 1.0])
        self.model = Convert(sp_net_config).convert(model)
        self.images = paddle.randn(shape=[2, 3, 224, 224], dtype='float32')
        self._test_clear_search_space()

    def _test_clear_search_space(self):
        self.ofa_model = OFA(self.model)
        self.ofa_model.set_epoch(0)
        outs = self.ofa_model(self.images)
        self.config = self.ofa_model.current_config

    def test_export_model(self):
        self.ofa_model.export(
            self.config,
            input_shapes=[[2, 3, 224, 224]],
            input_dtypes=['float32'])
        assert len(self.ofa_model.ofa_layers) == 37


class TestExportCase1(unittest.TestCase):
    def setUp(self):
        model = ModelLinear1()
        data_np = np.random.random((3, 64)).astype(np.int64)
        self.data = paddle.to_tensor(data_np)
        self.ofa_model = OFA(model)
        self.ofa_model.set_epoch(0)
        outs = self.ofa_model(self.data)
        self.config = self.ofa_model.current_config

    def test_export_model_linear1(self):
        ex_model = self.ofa_model.export(
            self.config, input_shapes=[[3, 64]], input_dtypes=['int64'])
        assert len(self.ofa_model.ofa_layers) == 3
        ex_model(self.data)


class TestExportCase2(unittest.TestCase):
    def setUp(self):
        model = ModelLinear()
        data_np = np.random.random((3, 64)).astype(np.int64)
        self.data = paddle.to_tensor(data_np)
        self.ofa_model = OFA(model)
        self.ofa_model.set_epoch(0)
        outs = self.ofa_model(self.data)
        self.config = self.ofa_model.current_config

    def test_export_model_linear2(self):
        config = self.ofa_model._sample_config(
            task='expand_ratio', phase=None, sample_type='smallest')
        ex_model = self.ofa_model.export(
            config, input_shapes=[[3, 64]], input_dtypes=['int64'])
        ex_model(self.data)
        assert len(self.ofa_model.ofa_layers) == 3


class TestManualSetting(unittest.TestCase):
    def setUp(self):
        self._init_model()

    def _init_model(self):
        model = ModelOriginLinear()
        data_np = np.random.random((3, 64)).astype(np.int64)
        self.data = paddle.to_tensor(data_np)
        sp_net_config = supernet(expand_ratio=[0.25, 0.5, 1.0])
        self.model = Convert(sp_net_config).convert(model)

    def test_setting_byhand(self):
        self.ofa_model1 = OFA(self.model)
        for key, value in self.ofa_model1._ofa_layers.items():
            if 'expand_ratio' in value:
                assert value['expand_ratio'] == [0.25, 0.5, 1.0]
        self.ofa_model1._clear_search_space(self.data)
        assert len(self.ofa_model1._ofa_layers) == 3

        ofa_layers = {
            'models.0': {
                'expand_ratio': [0.5, 1.0]
            },
            'models.1': {
                'expand_ratio': [0.25, 1.0]
            },
            'models.3': {
                'expand_ratio': [0.25, 1.0]
            },
            'models.4': {}
        }
        same_search_space = [['models.1', 'models.3']]
        skip_layers = ['models.0']
        cfg = {
            'ofa_layers': ofa_layers,
            'same_search_space': same_search_space,
            'skip_layers': skip_layers
        }
        run_config = RunConfig(**cfg)
        self.ofa_model2 = OFA(self.model, run_config=run_config)
        self.ofa_model2._clear_search_space(self.data)
        #print(self.ofa_model2._ofa_layers)
        assert self.ofa_model2._ofa_layers['models.1'][
            'expand_ratio'] == [0.25, 1.0]
        assert len(self.ofa_model2._ofa_layers) == 2
        #print(self.ofa_model_1._ofa_layers)

    def test_tokenize(self):
        self.ofa_model = OFA(self.model)
        self.ofa_model.set_task('expand_ratio')
        self.ofa_model._clear_search_space(self.data)
        self.ofa_model.tokenize()
        assert self.ofa_model.token_map == {
            'expand_ratio': {
                'models.0': {
                    0: 0.25,
                    1: 0.5,
                    2: 1.0
                },
                'models.1': {
                    0: 0.25,
                    1: 0.5,
                    2: 1.0
                },
                'models.3': {
                    0: 0.25,
                    1: 0.5,
                    2: 1.0
                }
            }
        }
        assert self.ofa_model.search_cands == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

        ofa_layers = {
            'models.0': {
                'expand_ratio': [0.5, 1.0]
            },
            'models.1': {
                'expand_ratio': [0.25, 1.0]
            },
            'models.3': {
                'expand_ratio': [0.25, 1.0]
            },
            'models.4': {}
        }
        same_search_space = [['models.1', 'models.3']]
        cfg = {'ofa_layers': ofa_layers, 'same_search_space': same_search_space}
        run_config = RunConfig(**cfg)
        self.ofa_model2 = OFA(self.model, run_config=run_config)
        self.ofa_model2.set_task('expand_ratio')
        self.ofa_model2._clear_search_space(self.data)
        self.ofa_model2.tokenize()
        assert self.ofa_model2.token_map == {
            'expand_ratio': {
                'models.0': {
                    1: 0.5,
                    2: 1.0
                },
                'models.1': {
                    0: 0.25,
                    2: 1.0
                }
            }
        }
        assert self.ofa_model2.search_cands == [[1, 2], [0, 2]]

        token = [1, 2]
        config = self.ofa_model2.decode_token(token)
        assert config == {'models.0': 0.5, 'models.1': 1.0}

    def test_input_spec(self):
        self.ofa_model = OFA(self.model)
        self.ofa_model.set_task('expand_ratio')
        self.ofa_model._clear_search_space(input_spec=[self.data])


if __name__ == '__main__':
    unittest.main()
