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

import os
import sys
import argparse
import functools
import logging
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from paddleslim.common import AvgrageMeter, get_logger
from paddleslim.dist import DML
from paddleslim.models.dygraph import MobileNetV1
import cifar100_reader as reader
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)
from utility import add_arguments, print_arguments

logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('log_freq',          int,   100,              "Log frequency.")
add_arg('batch_size',        int,   256,             "Minibatch size.")
add_arg('init_lr',           float, 0.1,             "The start learning rate.")
add_arg('use_gpu',           bool,  True,            "Whether use GPU.")
add_arg('epochs',            int,   200,             "Epoch number.")
add_arg('class_num',         int,   100,             "Class number of dataset.")
add_arg('trainset_num',      int,   50000,           "Images number of trainset.")
add_arg('model_save_dir',    str,   'saved_models',  "The path to save model.")
add_arg('use_multiprocess',  bool,  True,            "Whether use multiprocess reader.")
add_arg('use_parallel',      bool,  False,           "Whether to use data parallel mode to train the model.")
# yapf: enable


def create_optimizer(models, args):
    device_num = fluid.dygraph.parallel.Env().nranks
    step = int(args.trainset_num / (args.batch_size * device_num))
    epochs = [60, 120, 180]
    bd = [step * e for e in epochs]
    lr = [args.init_lr * (0.1**i) for i in range(len(bd) + 1)]

    optimizers = []
    for cur_model in models:
        learning_rate = fluid.dygraph.PiecewiseDecay(bd, lr, 0)
        opt = fluid.optimizer.MomentumOptimizer(
            learning_rate,
            0.9,
            parameter_list=cur_model.parameters(),
            use_nesterov=True,
            regularization=fluid.regularizer.L2DecayRegularizer(5e-4))
        optimizers.append(opt)
    return optimizers


def create_reader(place, args):
    train_reader = reader.train_valid(
        batch_size=args.batch_size, is_train=True, is_shuffle=True)
    valid_reader = reader.train_valid(
        batch_size=args.batch_size, is_train=False, is_shuffle=False)
    if args.use_parallel:
        train_reader = fluid.contrib.reader.distributed_batch_reader(
            train_reader)
    train_loader = fluid.io.DataLoader.from_generator(
        capacity=1024,
        return_list=True,
        use_multiprocess=args.use_multiprocess)
    valid_loader = fluid.io.DataLoader.from_generator(
        capacity=1024,
        return_list=True,
        use_multiprocess=args.use_multiprocess)
    train_loader.set_batch_generator(train_reader, places=place)
    valid_loader.set_batch_generator(valid_reader, places=place)
    return train_loader, valid_loader


def train(train_loader, dml_model, dml_optimizer, args):
    dml_model.train()
    costs = [AvgrageMeter() for i in range(dml_model.model_num)]
    accs = [AvgrageMeter() for i in range(dml_model.model_num)]
    for step_id, (images, labels) in enumerate(train_loader):
        images, labels = to_variable(images), to_variable(labels)
        batch_size = images.shape[0]

        logits = dml_model.forward(images)
        precs = [
            fluid.layers.accuracy(
                input=l, label=labels, k=1) for l in logits
        ]
        losses = dml_model.loss(logits, labels)
        dml_optimizer.minimize(losses)

        for i in range(dml_model.model_num):
            accs[i].update(precs[i].numpy(), batch_size)
            costs[i].update(losses[i].numpy(), batch_size)
        model_names = dml_model.full_name()
        if step_id % args.log_freq == 0:
            log_msg = "Train Step {}".format(step_id)
            for model_id, (cost, acc) in enumerate(zip(costs, accs)):
                log_msg += ", {} loss: {:.6f} acc: {:.6f}".format(
                    model_names[model_id], cost.avg[0], acc.avg[0])
            logger.info(log_msg)
    return costs, accs


def valid(valid_loader, dml_model, args):
    dml_model.eval()
    costs = [AvgrageMeter() for i in range(dml_model.model_num)]
    accs = [AvgrageMeter() for i in range(dml_model.model_num)]
    for step_id, (images, labels) in enumerate(valid_loader):
        images, labels = to_variable(images), to_variable(labels)
        batch_size = images.shape[0]

        logits = dml_model.forward(images)
        precs = [
            fluid.layers.accuracy(
                input=l, label=labels, k=1) for l in logits
        ]
        losses = dml_model.loss(logits, labels)

        for i in range(dml_model.model_num):
            accs[i].update(precs[i].numpy(), batch_size)
            costs[i].update(losses[i].numpy(), batch_size)
        model_names = dml_model.full_name()
        if step_id % args.log_freq == 0:
            log_msg = "Valid Step{} ".format(step_id)
            for model_id, (cost, acc) in enumerate(zip(costs, accs)):
                log_msg += ", {} loss: {:.6f} acc: {:.6f}".format(
                    model_names[model_id], cost.avg[0], acc.avg[0])
            logger.info(log_msg)
    return costs, accs


def main(args):
    if not args.use_gpu:
        place = fluid.CPUPlace()
    elif not args.use_parallel:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)

    with fluid.dygraph.guard(place):
        # 1. Define data reader
        train_loader, valid_loader = create_reader(place, args)

        # 2. Define neural network
        models = [
            MobileNetV1(class_dim=args.class_num),
            MobileNetV1(class_dim=args.class_num)
        ]
        optimizers = create_optimizer(models, args)

        # 3. Use PaddleSlim DML strategy
        dml_model = DML(models, args.use_parallel)
        dml_optimizer = dml_model.opt(optimizers)

        # 4. Train your network
        save_parameters = (not args.use_parallel) or (
            args.use_parallel and fluid.dygraph.parallel.Env().local_rank == 0)
        best_valid_acc = [0] * dml_model.model_num
        for epoch_id in range(args.epochs):
            current_step_lr = dml_optimizer.get_lr()
            lr_msg = "Epoch {}".format(epoch_id)
            for model_id, lr in enumerate(current_step_lr):
                lr_msg += ", {} lr: {:.6f}".format(
                    dml_model.full_name()[model_id], lr)
            logger.info(lr_msg)
            train_losses, train_accs = train(train_loader, dml_model,
                                             dml_optimizer, args)
            valid_losses, valid_accs = valid(valid_loader, dml_model, args)
            for i in range(dml_model.model_num):
                if valid_accs[i].avg[0] > best_valid_acc[i]:
                    best_valid_acc[i] = valid_accs[i].avg[0]
                    if save_parameters:
                        fluid.save_dygraph(
                            models[i].state_dict(),
                            os.path.join(args.model_save_dir,
                                         dml_model.full_name()[i],
                                         "best_model"))
                summery_msg = "Epoch {} {}: valid_loss {:.6f}, valid_acc {:.6f}, best_valid_acc {:.6f}"
                logger.info(
                    summery_msg.format(epoch_id,
                                       dml_model.full_name()[i], valid_losses[
                                           i].avg[0], valid_accs[i].avg[0],
                                       best_valid_acc[i]))


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    main(args)
