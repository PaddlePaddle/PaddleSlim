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

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn
from paddle.nn import ReLU
from paddleslim.nas.ofa import OFA, RunConfig, DistillConfig
from paddleslim.nas.ofa import supernet


class Model(fluid.dygraph.Layer):
    def __init__(self):
        super(Model, self).__init__()
        with supernet(
                kernel_size=(3, 5, 7),
                channel=((4, 8, 12), (8, 12, 16), (8, 12, 16))) as ofa_super:
            models = []
            models += [nn.Conv2D(3, 4, 3)]
            models += [nn.InstanceNorm(4)]
            models += [ReLU()]
            models += [nn.Conv2DTranspose(4, 4, 3, groups=4, use_cudnn=True)]
            models += [nn.BatchNorm(4)]
            models += [ReLU()]
            models += [
                fluid.dygraph.Pool2D(
                    pool_type='avg', global_pooling=True, use_cudnn=False)
            ]
            models += [nn.Conv2D(4, 3, 3)]
            models += [ReLU()]
            models = ofa_super.convert(models)
        self.models = paddle.nn.Sequential(*models)

    def forward(self, inputs, depth=None):
        print("depth: ", depth, len(self.models))
        if depth != None:
            assert isinstance(depth, int)
            assert depth < len(self.models)
        else:
            depth = len(self.models)
        # NOTE: fix slice model in sequential
        for idx in range(depth):
            inputs = self.models[idx](inputs)
        return inputs


def test_ofa():
    data_np = np.random.random((1, 3, 10, 10)).astype(np.float32)
    label_np = np.random.random((1)).astype(np.float32)

    default_run_config = {
        'train_batch_size': 1,  #256,
        'eval_batch_size': 1,  #64,
        'n_epochs': [[1], [2, 3], [4, 5]],
        'init_learning_rate': [[0.001], [0.003, 0.001], [0.003, 0.001]],
        'dynamic_batch_size': [1, 1, 1],
        'total_images': 1,  #1281167,
        'elastic_depth': (2, 5, 8)
    }
    run_config = RunConfig(**default_run_config)

    default_distill_config = {
        'lambda_distill': 0.01,
        'teacher_model': Model,
        'mapping_layers': ['models.0.fn']
    }
    distill_config = DistillConfig(**default_distill_config)

    fluid.enable_dygraph()
    model = Model()
    ofa_model = OFA(model, run_config, distill_config=distill_config)

    data = fluid.dygraph.to_variable(data_np)
    label = fluid.dygraph.to_variable(label_np)

    start_epoch = 0
    for idx in range(len(run_config.n_epochs)):
        cur_idx = run_config.n_epochs[idx]
        for ph_idx in range(len(cur_idx)):
            cur_lr = run_config.init_learning_rate[idx][ph_idx]
            adam = fluid.optimizer.Adam(
                learning_rate=cur_lr,
                parameter_list=(ofa_model.parameters() + ofa_model.netAs_param))
            for epoch_id in range(start_epoch,
                                  run_config.n_epochs[idx][ph_idx]):
                # add for data in dataset:
                for model_no in range(run_config.dynamic_batch_size[idx]):
                    output, _ = ofa_model(data)
                    print('current_config: ', ofa_model.current_config)
                    loss = fluid.layers.reduce_mean(output)
                    dis_loss = ofa_model.calc_distill_loss()
                    loss += dis_loss
                    print('epoch: {}, loss: {}, distill loss: {}'.format(
                        epoch_id, loss.numpy()[0], dis_loss.numpy()[0]))
                    loss.backward()
                    adam.minimize(loss)
                    adam.clear_gradients()
            start_epoch = run_config.n_epochs[idx][ph_idx]


test_ofa()
