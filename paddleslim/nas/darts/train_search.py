# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ['DARTSearch']

import math
import logging
from itertools import izip
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.framework import Variable
from paddle.fluid.dygraph.base import to_variable
from ...common import AvgrageMeter, get_logger
from .architect import Architect
logger = get_logger(__name__, level=logging.INFO)


def count_parameters_in_MB(all_params):
    parameters_number = 0
    for param in all_params:
        if param.trainable and 'aux' not in param.name:
            parameters_number += np.prod(param.shape)
    return parameters_number / 1e6


class DARTSearch(object):
    def __init__(self,
                 model,
                 train_reader,
                 valid_reader,
                 test_reader=None,
                 learning_rate=0.025,
                 batchsize=64,
                 num_imgs=50000,
                 arch_learning_rate=3e-4,
                 unrolled=False,
                 num_epochs=50,
                 epochs_no_archopt=0,
                 use_gpu=True,
                 use_data_parallel=False,
                 log_freq=50):
        self.model = model
        self.train_reader = train_reader
        self.valid_reader = valid_reader
        self.test_reader = test_reader
        self.learning_rate = learning_rate
        self.batchsize = batchsize
        self.num_imgs = num_imgs
        self.arch_learning_rate = arch_learning_rate
        self.unrolled = unrolled
        self.epochs_no_archopt = epochs_no_archopt
        self.num_epochs = num_epochs
        self.use_gpu = use_gpu
        self.use_data_parallel = use_data_parallel
        if not self.use_gpu:
            self.place = fluid.CPUPlace()
        elif not self.use_data_parallel:
            self.place = fluid.CUDAPlace(0)
        else:
            self.place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
        self.log_freq = log_freq

    def train_one_epoch(self, train_loader, valid_loader, architect, optimizer,
                        epoch):
        objs = AvgrageMeter()
        ce_losses = AvgrageMeter()
        kd_losses = AvgrageMeter()
        e_losses = AvgrageMeter()
        self.model.train()

        step_id = 0
        for train_data, valid_data in izip(train_loader(), valid_loader()):
            if epoch >= self.epochs_no_archopt:
                alphas_grad = architect.step(train_data, valid_data)

            loss, ce_loss, kd_loss, e_loss = self.model.loss(train_data)
            if math.isnan(e_loss.numpy()):
                print("alphas_grad: {}".format(alphas_grad))
                print("alphas: {}".format(self.model.arch_parameters()[0]
                                          .numpy()))
            if self.use_data_parallel:
                loss = self.model.scale_loss(loss)
                loss.backward()
                self.model.apply_collective_grads()
            else:
                loss.backward()

#            grad_clip = fluid.dygraph_grad_clip.GradClipByGlobalNorm(5)
#            optimizer.minimize(loss, grad_clip)
            optimizer.minimize(loss)

            self.model.clear_gradients()

            batch_size = train_data[0].shape[0]
            objs.update(loss.numpy(), batch_size)

            e_loss = e_loss.numpy() if isinstance(e_loss, Variable) else e_loss
            ce_losses.update(ce_loss.numpy(), batch_size)
            kd_losses.update(kd_loss.numpy(), batch_size)
            e_losses.update(e_loss, batch_size)

            if step_id % self.log_freq == 0:
                logger.info(
                    "Train Epoch {}, Step {}, loss {}; ce: {}; kd: {}; e: {}".
                    format(epoch, step_id,
                           loss.numpy(),
                           ce_loss.numpy(), kd_loss.numpy(), e_loss))
            step_id += 1
        return objs.avg[0]

    def valid_one_epoch(self, valid_loader, epoch):
        self.model.eval()
        meters = {}
        for step_id, valid_data in enumerate(valid_loader):
            ret = self.model.valid(valid_data)
            for key, value in ret.items():
                if key not in meters:
                    meters[key] = AvgrageMeter()
                meters[key].update(value, 1)

            if step_id % self.log_freq == 0:
                logger.info("Valid Epoch {}, Step {}, {}".format(
                    epoch, step_id, meters))

    def train(self):
        if self.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()
        model_parameters = self.model.model_parameters()
        logger.info("parameter size in super net: {:.6f}M".format(
            count_parameters_in_MB(model_parameters)))
        step_per_epoch = int(self.num_imgs * 0.5 / self.batchsize)
        if self.unrolled:
            step_per_epoch *= 2
        learning_rate = fluid.dygraph.CosineDecay(
            self.learning_rate, step_per_epoch, self.num_epochs)

        optimizer = fluid.optimizer.MomentumOptimizer(
            learning_rate,
            0.9,
            regularization=fluid.regularizer.L2DecayRegularizer(3e-4),
            parameter_list=model_parameters)

        if self.use_data_parallel:
            self.model = fluid.dygraph.parallel.DataParallel(self.model,
                                                             strategy)
            self.train_reader = fluid.contrib.reader.distributed_batch_reader(
                self.train_reader)
            self.valid_reader = fluid.contrib.reader.distributed_batch_reader(
                self.valid_reader)
            if self.test_reader is not None:
                self.test_reader = fluid.contrib.reader.distributed_batch_reader(
                    self.test_reader)

        train_loader = fluid.io.DataLoader.from_generator(
            capacity=64,
            use_double_buffer=True,
            iterable=True,
            return_list=True)
        valid_loader = fluid.io.DataLoader.from_generator(
            capacity=64,
            use_double_buffer=True,
            iterable=True,
            return_list=True)

        train_loader.set_batch_generator(self.train_reader, places=self.place)
        valid_loader.set_batch_generator(self.valid_reader, places=self.place)

        if self.test_reader is not None:
            test_loader = fluid.io.DataLoader.from_generator(
                capacity=64,
                use_double_buffer=True,
                iterable=True,
                return_list=True)
            test_loader.set_batch_generator(
                self.test_reader, places=self.place)
        else:
            test_loader = valid_loader

        architect = Architect(self.model, learning_rate,
                              self.arch_learning_rate, self.place,
                              self.unrolled)

        save_parameters = (not self.use_data_parallel) or (
            self.use_data_parallel and
            fluid.dygraph.parallel.Env().local_rank == 0)

        for epoch in range(self.num_epochs):
            logger.info('Epoch {}, lr {:.6f}'.format(
                epoch, optimizer.current_step_lr()))
            genotype = self.model.genotype()
            logger.info('genotype = %s', genotype)

            self.train_one_epoch(train_loader, valid_loader, architect,
                                 optimizer, epoch)


#            if epoch == self.num_epochs - 1:
#                self.valid_one_epoch(test_loader, epoch)
#            if save_parameters:
#                fluid.save_dygraph(self.model.state_dict(), "./weights")
