# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append("../")
import unittest
import logging
import numpy as np
import paddle
from static_case import StaticCase
import paddle.fluid as fluid
import paddle.dataset.mnist as reader
from paddle.fluid.dygraph.base import to_variable
from paddleslim.models.dygraph import MobileNetV1
from paddleslim.dist import DML
from paddleslim.common import get_logger
logger = get_logger(__name__, level=logging.INFO)


class Model(fluid.dygraph.Layer):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = fluid.dygraph.nn.Conv2D(
            num_channels=1,
            num_filters=256,
            filter_size=3,
            stride=1,
            padding=1,
            use_cudnn=False)
        self.pool2d_avg = fluid.dygraph.nn.Pool2D(
            pool_type='avg', global_pooling=True)
        self.out = fluid.dygraph.nn.Linear(256, 10)

    def forward(self, inputs):
        inputs = fluid.layers.reshape(inputs, shape=[0, 1, 28, 28])
        y = self.conv(inputs)
        y = self.pool2d_avg(y)
        y = fluid.layers.reshape(y, shape=[-1, 256])
        y = self.out(y)
        return y


class TestDML(StaticCase):
    def test_dml(self):
        place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            train_reader = paddle.fluid.io.batch(
                paddle.dataset.mnist.train(), batch_size=256)
            train_loader = fluid.io.DataLoader.from_generator(
                capacity=1024, return_list=True)
            train_loader.set_sample_list_generator(train_reader, places=place)

            models = [Model(), Model()]
            optimizers = []
            for cur_model in models:
                opt = fluid.optimizer.MomentumOptimizer(
                    0.1, 0.9, parameter_list=cur_model.parameters())
                optimizers.append(opt)
            dml_model = DML(models)
            dml_optimizer = dml_model.opt(optimizers)

            def train(train_loader, dml_model, dml_optimizer):
                dml_model.train()
                for step_id, (images, labels) in enumerate(train_loader):
                    images, labels = to_variable(images), to_variable(labels)
                    labels = fluid.layers.reshape(labels, [0, 1])

                    logits = dml_model.forward(images)
                    precs = [
                        fluid.layers.accuracy(
                            input=l, label=labels, k=1).numpy() for l in logits
                    ]
                    losses = dml_model.loss(logits, labels)
                    dml_optimizer.minimize(losses)
                    if step_id % 10 == 0:
                        print(step_id, precs)

            for epoch_id in range(10):
                current_step_lr = dml_optimizer.get_lr()
                lr_msg = "Epoch {}".format(epoch_id)
                for model_id, lr in enumerate(current_step_lr):
                    lr_msg += ", {} lr: {:.6f}".format(
                        dml_model.full_name()[model_id], lr)
                logger.info(lr_msg)
                train(train_loader, dml_model, dml_optimizer)


if __name__ == '__main__':
    unittest.main()
