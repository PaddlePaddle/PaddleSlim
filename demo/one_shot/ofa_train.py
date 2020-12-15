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
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import ReLU
from paddleslim.nas.ofa import OFA, RunConfig, DistillConfig
from paddleslim.nas.ofa import supernet


class Model(nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        with supernet(
                kernel_size=(3, 5, 7), expand_ratio=[1, 2, 4]) as ofa_super:
            models = []
            models += [nn.Conv2D(1, 6, 3)]
            models += [ReLU()]
            models += [nn.Pool2D(2, 'max', 2)]
            models += [nn.Conv2D(6, 16, 5, padding=0)]
            models += [ReLU()]
            models += [nn.Pool2D(2, 'max', 2)]
            models += [
                nn.Linear(784, 120), nn.Linear(120, 84), nn.Linear(84, 10)
            ]
            models = ofa_super.convert(models)
        self.models = paddle.nn.Sequential(*models)

    def forward(self, inputs, label, depth=None):
        if depth != None:
            assert isinstance(depth, int)
            assert depth < len(self.models)
            models = self.models[:depth]
        else:
            depth = len(self.models)
            models = self.models[:]

        for idx, layer in enumerate(models):
            if idx == 6:
                inputs = paddle.flatten(inputs, 1)
            inputs = layer(inputs)

        inputs = F.softmax(inputs)
        return inputs


def test_ofa():

    model = Model()
    teacher_model = Model()

    default_run_config = {
        'train_batch_size': 256,
        'n_epochs': [[1], [2, 3], [4, 5]],
        'init_learning_rate': [[0.001], [0.003, 0.001], [0.003, 0.001]],
        'dynamic_batch_size': [1, 1, 1],
        'total_images': 50000,  #1281167,
        'elastic_depth': (2, 5, 8)
    }
    run_config = RunConfig(**default_run_config)

    default_distill_config = {
        'lambda_distill': 0.01,
        'teacher_model': teacher_model,
        'mapping_layers': ['models.0.fn']
    }
    distill_config = DistillConfig(**default_distill_config)

    ofa_model = OFA(model, run_config, distill_config=distill_config)

    train_dataset = paddle.vision.datasets.MNIST(
        mode='train', backend='cv2', transform=transform)
    train_loader = paddle.io.DataLoader(
        train_dataset,
        places=place,
        feed_list=[image, label],
        drop_last=True,
        batch_size=64)

    start_epoch = 0
    for idx in range(len(run_config.n_epochs)):
        cur_idx = run_config.n_epochs[idx]
        for ph_idx in range(len(cur_idx)):
            cur_lr = run_config.init_learning_rate[idx][ph_idx]
            adam = paddle.optimizer.Adam(
                learning_rate=cur_lr,
                parameter_list=(ofa_model.parameters() + ofa_model.netAs_param))
            for epoch_id in range(start_epoch,
                                  run_config.n_epochs[idx][ph_idx]):
                for batch_id, data in enumerate(train_loader()):
                    dy_x_data = np.array(
                        [x[0].reshape(1, 28, 28)
                         for x in data]).astype('float32')
                    y_data = np.array(
                        [x[1] for x in data]).astype('int64').reshape(-1, 1)

                    img = paddle.dygraph.to_variable(dy_x_data)
                    label = paddle.dygraph.to_variable(y_data)
                    label.stop_gradient = True

                    for model_no in range(run_config.dynamic_batch_size[idx]):
                        output, _ = ofa_model(img, label)
                        loss = F.mean(output)
                        dis_loss = ofa_model.calc_distill_loss()
                        loss += dis_loss
                        loss.backward()

                        if batch_id % 10 == 0:
                            print(
                                'epoch: {}, batch: {}, loss: {}, distill loss: {}'.
                                format(epoch_id, batch_id,
                                       loss.numpy()[0], dis_loss.numpy()[0]))
                    ### accumurate dynamic_batch_size network of gradients for same batch of data
                    ### NOTE: need to fix gradients accumulate in PaddlePaddle
                    adam.minimize(loss)
                    adam.clear_gradients()
            start_epoch = run_config.n_epochs[idx][ph_idx]


test_ofa()
