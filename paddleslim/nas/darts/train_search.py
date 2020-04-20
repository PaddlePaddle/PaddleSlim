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

import logging
from itertools import izip
import numpy as np
import paddle.fluid as fluid
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
                architect.step(train_data, valid_data)

            loss, ce_loss, kd_loss, e_loss = self.model.loss(train_data)

            if self.use_data_parallel:
                loss = self.model.scale_loss(loss)
                loss.backward()
                self.model.apply_collective_grads()
            else:
                loss.backward()

            grad_clip = fluid.dygraph_grad_clip.GradClipByGlobalNorm(5)
            optimizer.minimize(loss, grad_clip)
            self.model.clear_gradients()

            batch_size = train_data[0].shape[0]
            objs.update(loss.numpy(), batch_size)
            ce_losses.update(ce_loss.numpy(), batch_size)
            kd_losses.update(kd_loss.numpy(), batch_size)
            e_losses.update(e_loss.numpy(), batch_size)

            if step_id % self.log_freq == 0:
                #logger.info("Train Epoch {}, Step {}, loss {:.6f}; ce: {:.6f}; kd: {:.6f}; e: {:.6f}".format(
                #    epoch, step_id, objs.avg[0], ce_losses.avg[0], kd_losses.avg[0], e_losses.avg[0]))
                logger.info(
                    "Train Epoch {}, Step {}, loss {}; ce: {}; kd: {}; e: {}".
                    format(epoch, step_id,
                           loss.numpy(),
                           ce_loss.numpy(), kd_loss.numpy(), e_loss.numpy()))
            step_id += 1
        return objs.avg[0]

    def valid_one_epoch(self, valid_loader, epoch):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()
        self.model.eval()

        for step_id, valid_data in enumerate(valid_loader):
            image = to_variable(image)
            label = to_variable(label)
            n = image.shape[0]
            logits = self.model(image)
            prec1 = fluid.layers.accuracy(input=logits, label=label, k=1)
            prec5 = fluid.layers.accuracy(input=logits, label=label, k=5)
            loss = fluid.layers.reduce_mean(
                fluid.layers.softmax_with_cross_entropy(logits, label))
            objs.update(loss.numpy(), n)
            top1.update(prec1.numpy(), n)
            top5.update(prec5.numpy(), n)

            if step_id % self.log_freq == 0:
                logger.info(
                    "Valid Epoch {}, Step {}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".
                    format(epoch, step_id, objs.avg[0], top1.avg[0], top5.avg[
                        0]))
        return top1.avg[0]

    def train(self):
        if self.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()
        model_parameters = [
            p for p in self.model.parameters()
            if p.name not in [a.name for a in self.model.arch_parameters()]
        ]
        logger.info("param size = {:.6f}MB".format(
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

            if epoch == self.num_epochs - 1:
                #                valid_top1 = self.valid_one_epoch(valid_loader, epoch)
                logger.info("Epoch {}, valid_acc {:.6f}".format(epoch, 1))
            if save_parameters:
                fluid.save_dygraph(self.model.state_dict(), "./weights")
