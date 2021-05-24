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

from __future__ import print_function
import argparse
import ast
import numpy as np
from PIL import Image
import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from paddle.fluid.dygraph.base import to_variable

from paddleslim.nas.one_shot import SuperMnasnet
from paddleslim.nas.one_shot import OneShotSearch


def parse_args():
    parser = argparse.ArgumentParser("Training for Mnist.")
    parser.add_argument(
        "--use_data_parallel",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to use data parallel mode to train the model."
    )
    parser.add_argument("-e", "--epoch", default=5, type=int, help="set epoch")
    parser.add_argument("--ce", action="store_true", help="run ce")
    args = parser.parse_args()
    return args


class SimpleImgConv(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 conv_stride=1,
                 conv_padding=0,
                 conv_dilation=1,
                 conv_groups=1,
                 act=None,
                 use_cudnn=False,
                 param_attr=None,
                 bias_attr=None):
        super(SimpleImgConv, self).__init__()

        self._conv2d = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            param_attr=None,
            bias_attr=None,
            act=act,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        return x


class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        self._simple_img_conv_pool_1 = SimpleImgConv(1, 20, 2, act="relu")
        self.arch = SuperMnasnet(
            name_scope="super_net", input_channels=20, out_channels=20)
        self._simple_img_conv_pool_2 = SimpleImgConv(20, 50, 2, act="relu")

        self.pool_2_shape = 50 * 13 * 13
        SIZE = 10
        scale = (2.0 / (self.pool_2_shape**2 * SIZE))**0.5
        self._fc = Linear(
            self.pool_2_shape,
            10,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=scale)),
            act="softmax")

    def forward(self, inputs, label=None, tokens=None):
        x = self._simple_img_conv_pool_1(inputs)

        x = self.arch(x, tokens=tokens)  # addddddd
        x = self._simple_img_conv_pool_2(x)
        x = fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])
        x = self._fc(x)
        if label is not None:
            acc = fluid.layers.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


def test_mnist(model, tokens=None):
    acc_set = []
    avg_loss_set = []
    batch_size = 64
    test_reader = paddle.fluid.io.batch(
        paddle.dataset.mnist.test(), batch_size=batch_size, drop_last=True)
    for batch_id, data in enumerate(test_reader()):
        dy_x_data = np.array([x[0].reshape(1, 28, 28)
                              for x in data]).astype('float32')
        y_data = np.array(
            [x[1] for x in data]).astype('int64').reshape(batch_size, 1)

        img = to_variable(dy_x_data)
        label = to_variable(y_data)
        label.stop_gradient = True
        prediction, acc = model.forward(img, label, tokens=tokens)
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_loss = fluid.layers.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))
        if batch_id % 100 == 0:
            print("Test - batch_id: {}".format(batch_id))
        # get test acc and loss
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    return acc_val_mean


def train_mnist(args, model, tokens=None):
    epoch_num = args.epoch
    BATCH_SIZE = 64

    adam = AdamOptimizer(
        learning_rate=0.001, parameter_list=model.parameters())

    train_reader = paddle.fluid.io.batch(
        paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
    if args.use_data_parallel:
        train_reader = fluid.contrib.reader.distributed_batch_reader(
            train_reader)

    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_reader()):
            dy_x_data = np.array([x[0].reshape(1, 28, 28)
                                  for x in data]).astype('float32')
            y_data = np.array(
                [x[1] for x in data]).astype('int64').reshape(-1, 1)

            img = to_variable(dy_x_data)
            label = to_variable(y_data)
            label.stop_gradient = True

            cost, acc = model.forward(img, label, tokens=tokens)

            loss = fluid.layers.cross_entropy(cost, label)
            avg_loss = fluid.layers.mean(loss)

            if args.use_data_parallel:
                avg_loss = model.scale_loss(avg_loss)
                avg_loss.backward()
                model.apply_collective_grads()
            else:
                avg_loss.backward()

            adam.minimize(avg_loss)
            # save checkpoint
            model.clear_gradients()
            if batch_id % 1 == 0:
                print("Loss at epoch {} step {}: {:}".format(epoch, batch_id,
                                                             avg_loss.numpy()))

        model.eval()
        test_acc = test_mnist(model, tokens=tokens)
        model.train()
        print("Loss at epoch {} , acc is: {}".format(epoch, test_acc))

    save_parameters = (not args.use_data_parallel) or (
        args.use_data_parallel and
        fluid.dygraph.parallel.Env().local_rank == 0)
    if save_parameters:
        fluid.save_dygraph(model.state_dict(), "save_temp")
        print("checkpoint saved")


if __name__ == '__main__':
    args = parse_args()
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = MNIST()
        # step 1: training super net
        #train_mnist(args, model)
        # step 2: search
        best_tokens = OneShotSearch(model, test_mnist)
    # step 3: final training
    #    train_mnist(args, model, best_tokens)
