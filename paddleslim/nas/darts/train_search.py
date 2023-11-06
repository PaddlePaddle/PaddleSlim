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
import paddle

__all__ = ['DARTSearch', 'count_parameters_in_MB']

import os
import logging
import numpy as np
from ...common import AvgrageMeter, get_logger
from .architect import Architect
from .get_genotype import get_genotype
logger = get_logger(__name__, level=logging.INFO)


def count_parameters_in_MB(all_params):
    """Count the parameters in the target list.
    Args:
        all_params(list): List of Variables.

    Returns:
        float: The total count(MB) of target parameter list.
    """

    parameters_number = 0
    for param in all_params:
        if param.trainable and 'aux' not in param.name:
            parameters_number += np.prod(param.shape)
    return parameters_number / 1e6


class DARTSearch(object):
    """Used for Differentiable ARchiTecture Search(DARTS)

    Args:
        model(Paddle DyGraph model): Super Network for Search.
        train_reader(Python Generator): Generator to provide training data.
        valid_reader(Python Generator): Generator to provide validation  data.
        place(paddle.CPUPlace()|paddle.CUDAPlace(N)): This parameter represents the executor run on which device.
        learning_rate(float): Model parameter initial learning rate. Default: 0.025.
        batch_size(int): Minibatch size. Default: 64.
        arch_learning_rate(float): Learning rate for arch encoding. Default: 3e-4.
        unrolled(bool): Use one-step unrolled validation loss. Default: False.
        num_epochs(int): Epoch number. Default: 50.
        epochs_no_archopt(int): Epochs skip architecture optimize at begining. Default: 0.
        use_multiprocess(bool): Whether to use multiprocess in dataloader. Default: False.
        log_freq(int): Log frequency. Default: 50.

    """

    def __init__(self,
                 model,
                 train_reader,
                 valid_reader,
                 place,
                 learning_rate=0.025,
                 batchsize=64,
                 num_imgs=50000,
                 arch_learning_rate=3e-4,
                 unrolled=False,
                 num_epochs=50,
                 epochs_no_archopt=0,
                 use_multiprocess=False,
                 save_dir='./',
                 log_freq=50):
        self.model = model
        self.train_reader = train_reader
        self.valid_reader = valid_reader
        self.place = place,
        self.learning_rate = learning_rate
        self.batchsize = batchsize
        self.num_imgs = num_imgs
        self.arch_learning_rate = arch_learning_rate
        self.unrolled = unrolled
        self.epochs_no_archopt = epochs_no_archopt
        self.num_epochs = num_epochs
        self.use_multiprocess = use_multiprocess
        self.save_dir = save_dir
        self.log_freq = log_freq

    def train_one_epoch(self, train_loader, valid_loader, architect, optimizer,
                        epoch):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()
        self.model.train()

        for step_id, (
                train_data,
                valid_data) in enumerate(zip(train_loader(), valid_loader())):
            train_image, train_label = train_data
            valid_image, valid_label = valid_data
            train_image = paddle.to_tensor(data=train_image)
            train_label = paddle.to_tensor(data=train_label)
            train_label.stop_gradient = True
            valid_image = paddle.to_tensor(data=valid_image)
            valid_label = paddle.to_tensor(data=valid_label)
            valid_label.stop_gradient = True
            n = train_image.shape[0]

            if epoch >= self.epochs_no_archopt:
                architect.step(train_image, train_label, valid_image,
                               valid_label)

            logits = self.model(train_image)
            prec1 = paddle.static.accuracy(input=logits, label=train_label, k=1)
            prec5 = paddle.static.accuracy(input=logits, label=train_label, k=5)
            loss = paddle.mean(
                x=paddle.nn.functional.softmax_with_cross_entropy(
                    logits, train_label))

            loss.backward()

            optimizer.minimize(loss)
            self.model.clear_gradients()

            objs.update(loss.numpy(), n)
            top1.update(prec1.numpy(), n)
            top5.update(prec5.numpy(), n)

            if step_id % self.log_freq == 0:
                #logger.info("Train Epoch {}, Step {}, loss {:.6f}; ce: {:.6f}; kd: {:.6f}; e: {:.6f}".format(
                #    epoch, step_id, objs.avg[0], ce_losses.avg[0], kd_losses.avg[0], e_losses.avg[0]))
                logger.info(
                    "Train Epoch {}, Step {}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".
                    format(epoch, step_id, objs.avg[0], top1.avg[0], top5.avg[
                        0]))
        return top1.avg[0]

    def valid_one_epoch(self, valid_loader, epoch):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()
        self.model.eval()

        for step_id, (image, label) in enumerate(valid_loader):
            image = paddle.to_tensor(data=image)
            label = paddle.to_tensor(data=label)
            n = image.shape[0]
            logits = self.model(image)
            prec1 = paddle.static.accuracy(input=logits, label=label, k=1)
            prec5 = paddle.static.accuracy(input=logits, label=label, k=5)
            loss = paddle.mean(
                x=paddle.nn.functional.softmax_with_cross_entropy(
                    logits, label))
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
        """Start search process.

        """

        model_parameters = [
            p for p in self.model.parameters()
            if p.name not in [a.name for a in self.model.arch_parameters()]
        ]
        logger.info("param size = {:.6f}MB".format(
            count_parameters_in_MB(model_parameters)))

        device_num = 1
        step_per_epoch = int(self.num_imgs * 0.5 /
                             (self.batchsize * device_num))
        if self.unrolled:
            step_per_epoch *= 2
        learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
            self.learning_rate, self.num_epochs // 2)

        clip = paddle.nn.ClipGradByGlobalNorm(5.0)
        optimizer = paddle.optimizer.Momentum(
            learning_rate,
            0.9,
            weight_decay=3e-4,
            parameters=model_parameters,
            grad_clip=clip)

        train_loader = paddle.io.DataLoader(
            self.train_reader, use_buffer_reader=True, return_list=True)
        valid_loader = paddle.io.DataLoader(
            self.valid_reader, use_buffer_reader=True, return_list=True)

        base_model = self.model
        architect = Architect(
            model=self.model,
            eta=learning_rate,
            arch_learning_rate=self.arch_learning_rate,
            unrolled=self.unrolled)

        self.model = architect.get_model()

        save_parameters = True

        for epoch in range(self.num_epochs):
            logger.info('Epoch {}, lr {:.6f}'.format(epoch, optimizer.get_lr()))

            genotype = get_genotype(base_model)
            logger.info('genotype = %s', genotype)

            train_top1 = self.train_one_epoch(train_loader, valid_loader,
                                              architect, optimizer, epoch)
            logger.info("Epoch {}, train_acc {:.6f}".format(epoch, train_top1))

            if epoch == self.num_epochs - 1:
                valid_top1 = self.valid_one_epoch(valid_loader, epoch)
                logger.info(
                    "Epoch {}, valid_acc {:.6f}".format(epoch, valid_top1))
            if save_parameters:
                paddle.save(self.model.state_dict(),
                            os.path.join(self.save_dir, str(epoch), "params"))
