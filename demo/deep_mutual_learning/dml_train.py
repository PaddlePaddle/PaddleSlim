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
import paddle
from paddleslim.common import AvgrageMeter, get_logger
from paddleslim.dist import DML
from paddleslim.models.dygraph import MobileNetV1
from paddleslim.models.dygraph import ResNet
import cifar100_reader as reader
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)
from utility import add_arguments, print_arguments

logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('log_freq',          int,   100,              "Log frequency.")
add_arg('models',            str,   "mobilenet-mobilenet",  "model.")
add_arg('batch_size',        int,   256,             "Minibatch size.")
add_arg('init_lr',           float, 0.1,             "The start learning rate.")
add_arg('use_gpu',           bool,  True,            "Whether use GPU.")
add_arg('epochs',            int,   200,             "Epoch number.")
add_arg('class_num',         int,   100,             "Class number of dataset.")
add_arg('trainset_num',      int,   50000,           "Images number of trainset.")
add_arg('model_save_dir',    str,   'saved_models',  "The path to save model.")
# yapf: enable


def create_optimizer(models, args):
    step = int(args.trainset_num / (args.batch_size))
    epochs = [60, 120, 180]
    bd = [step * e for e in epochs]
    lr = [args.init_lr * (0.1**i) for i in range(len(bd) + 1)]

    optimizers = []
    for cur_model in models:
        learning_rate = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=bd, values=lr)
        opt = paddle.optimizer.Momentum(
            learning_rate,
            0.9,
            parameters=cur_model.parameters(),
            use_nesterov=True,
            weight_decay=paddle.regularizer.L2Decay(5e-4))
        optimizers.append(opt)
    return optimizers, learning_rate


def create_reader(place, args):
    train_reader = reader.train_valid(
        batch_size=args.batch_size, is_train=True, is_shuffle=True)
    valid_reader = reader.train_valid(
        batch_size=args.batch_size, is_train=False, is_shuffle=False)
    train_loader = paddle.io.DataLoader.from_generator(
        capacity=1024, return_list=True)
    valid_loader = paddle.io.DataLoader.from_generator(
        capacity=1024, return_list=True)
    train_loader.set_batch_generator(train_reader, places=place)
    valid_loader.set_batch_generator(valid_reader, places=place)
    return train_loader, valid_loader


def train(train_loader, dml_model, dml_optimizer, lr, args):
    dml_model.train()
    costs = [AvgrageMeter() for i in range(dml_model.model_num)]
    accs = [AvgrageMeter() for i in range(dml_model.model_num)]
    for step_id, (images, labels) in enumerate(train_loader):
        images, labels = paddle.to_tensor(images), paddle.to_tensor(labels)
        batch_size = images.shape[0]

        logits = dml_model.forward(images)
        precs = [
            paddle.metric.accuracy(
                input=l, label=labels, k=1) for l in logits
        ]
        losses = dml_model.loss(logits, labels)
        dml_optimizer.minimize(losses)
        lr.step()

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
        images, labels = paddle.to_tensor(images), paddle.to_tensor(labels)
        batch_size = images.shape[0]

        logits = dml_model.forward(images)
        precs = [
            paddle.metric.accuracy(
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
        place = paddle.CPUPlace()
    else:
        place = paddle.CUDAPlace(0)

    # 1. Define data reader
    train_loader, valid_loader = create_reader(place, args)

    # 2. Define neural network
    if args.models == "mobilenet-mobilenet":
        models = [
            MobileNetV1(class_dim=args.class_num),
            MobileNetV1(class_dim=args.class_num)
        ]
    elif args.models == "mobilenet-resnet50":
        models = [
            MobileNetV1(class_dim=args.class_num),
            ResNet(class_dim=args.class_num)
        ]
    else:
        logger.info("You can define the model as you wish")
        return
    optimizers, lr = create_optimizer(models, args)

    # 3. Use PaddleSlim DML strategy
    dml_model = DML(models)
    dml_optimizer = dml_model.opt(optimizers)

    # 4. Train your network
    best_valid_acc = [0] * dml_model.model_num
    for epoch_id in range(args.epochs):
        train_losses, train_accs = train(train_loader, dml_model, dml_optimizer,
                                         lr, args)
        valid_losses, valid_accs = valid(valid_loader, dml_model, args)
        for i in range(dml_model.model_num):
            if valid_accs[i].avg[0] > best_valid_acc[i]:
                best_valid_acc[i] = valid_accs[i].avg[0]
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
