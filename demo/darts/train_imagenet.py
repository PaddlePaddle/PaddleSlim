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
import ast
import logging
import argparse
import functools

import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from paddleslim.common import AvgrageMeter, get_logger
from paddleslim.nas.darts import count_parameters_in_MB

import genotypes
import reader
from model import NetworkImageNet as Network
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)
from utility import add_arguments, print_arguments
logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('use_multiprocess',  bool,  True,            "Whether use multiprocess reader.")
add_arg('num_workers',       int,   4,               "The multiprocess reader number.")
add_arg('data_dir',          str,   'dataset/ILSVRC2012',"The dir of dataset.")
add_arg('batch_size',        int,   128,             "Minibatch size.")
add_arg('learning_rate',     float, 0.1,             "The start learning rate.")
add_arg('decay_rate',        float, 0.97,            "The lr decay rate.")
add_arg('momentum',          float, 0.9,             "Momentum.")
add_arg('weight_decay',      float, 3e-5,            "Weight_decay.")
add_arg('use_gpu',           bool,  True,            "Whether use GPU.")
add_arg('epochs',            int,   250,             "Epoch number.")
add_arg('init_channels',     int,   48,              "Init channel number.")
add_arg('layers',            int,   14,              "Total number of layers.")
add_arg('class_num',         int,   1000,            "Class number of dataset.")
add_arg('trainset_num',      int,   1281167,         "Images number of trainset.")
add_arg('model_save_dir',    str,   'eval_imagenet', "The path to save model.")
add_arg('auxiliary',         bool,  True,            'Use auxiliary tower.')
add_arg('auxiliary_weight',  float, 0.4,             "Weight for auxiliary loss.")
add_arg('drop_path_prob',    float, 0.0,             "Drop path probability.")
add_arg('dropout',           float, 0.0,             "Dropout probability.")
add_arg('grad_clip',         float, 5,               "Gradient clipping.")
add_arg('label_smooth',      float, 0.1,             "Label smoothing.")
add_arg('arch',              str,   'DARTS_V2',      "Which architecture to use")
add_arg('log_freq',          int,   100,             'Report frequency')
add_arg('use_data_parallel', ast.literal_eval,  False, "The flag indicating whether to use data parallel mode to train the model.")
# yapf: enable


def cross_entropy_label_smooth(preds, targets, epsilon):
    preds = fluid.layers.softmax(preds)
    targets_one_hot = fluid.one_hot(input=targets, depth=args.class_num)
    targets_smooth = fluid.layers.label_smooth(
        targets_one_hot, epsilon=epsilon, dtype="float32")
    loss = fluid.layers.cross_entropy(
        input=preds, label=targets_smooth, soft_label=True)
    return loss


def train(model, train_reader, optimizer, epoch, args):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.train()

    for step_id, data in enumerate(train_reader()):
        image_np, label_np = data
        image = to_variable(image_np)
        label = to_variable(label_np)
        label.stop_gradient = True
        logits, logits_aux = model(image, True)

        prec1 = fluid.layers.accuracy(input=logits, label=label, k=1)
        prec5 = fluid.layers.accuracy(input=logits, label=label, k=5)
        loss = fluid.layers.reduce_mean(
            cross_entropy_label_smooth(logits, label, args.label_smooth))

        if args.auxiliary:
            loss_aux = fluid.layers.reduce_mean(
                cross_entropy_label_smooth(logits_aux, label,
                                           args.label_smooth))
            loss = loss + args.auxiliary_weight * loss_aux

        if args.use_data_parallel:
            loss = model.scale_loss(loss)
            loss.backward()
            model.apply_collective_grads()
        else:
            loss.backward()

        optimizer.minimize(loss)
        model.clear_gradients()

        n = image.shape[0]
        objs.update(loss.numpy(), n)
        top1.update(prec1.numpy(), n)
        top5.update(prec5.numpy(), n)

        if step_id % args.log_freq == 0:
            logger.info(
                "Train Epoch {}, Step {}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".
                format(epoch, step_id, objs.avg[0], top1.avg[0], top5.avg[0]))
    return top1.avg[0], top5.avg[0]


def valid(model, valid_reader, epoch, args):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()

    for step_id, data in enumerate(valid_reader()):
        image_np, label_np = data
        image = to_variable(image_np)
        label = to_variable(label_np)
        logits, _ = model(image, False)
        prec1 = fluid.layers.accuracy(input=logits, label=label, k=1)
        prec5 = fluid.layers.accuracy(input=logits, label=label, k=5)
        loss = fluid.layers.reduce_mean(
            cross_entropy_label_smooth(logits, label, args.label_smooth))

        n = image.shape[0]
        objs.update(loss.numpy(), n)
        top1.update(prec1.numpy(), n)
        top5.update(prec5.numpy(), n)
        if step_id % args.log_freq == 0:
            logger.info(
                "Valid Epoch {}, Step {}, loss {:.6f}, acc_1 {:.6f}, acc_5 {:.6f}".
                format(epoch, step_id, objs.avg[0], top1.avg[0], top5.avg[0]))
    return top1.avg[0], top5.avg[0]


def main(args):
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if args.use_data_parallel else fluid.CUDAPlace(0)

    with fluid.dygraph.guard(place):
        genotype = eval("genotypes.%s" % args.arch)
        model = Network(
            C=args.init_channels,
            num_classes=args.class_num,
            layers=args.layers,
            auxiliary=args.auxiliary,
            genotype=genotype)

        logger.info("param size = {:.6f}MB".format(
            count_parameters_in_MB(model.parameters())))

        device_num = fluid.dygraph.parallel.Env().nranks
        step_per_epoch = int(args.trainset_num /
                             (args.batch_size * device_num))
        learning_rate = fluid.dygraph.ExponentialDecay(
            args.learning_rate,
            step_per_epoch,
            args.decay_rate,
            staircase=True)

        clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=args.grad_clip)
        optimizer = fluid.optimizer.MomentumOptimizer(
            learning_rate,
            momentum=args.momentum,
            regularization=fluid.regularizer.L2Decay(args.weight_decay),
            parameter_list=model.parameters(),
            grad_clip=clip)

        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()
            model = fluid.dygraph.parallel.DataParallel(model, strategy)

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

        train_reader = fluid.io.batch(
            reader.imagenet_reader(args.data_dir, 'train'),
            batch_size=args.batch_size,
            drop_last=True)
        valid_reader = fluid.io.batch(
            reader.imagenet_reader(args.data_dir, 'val'),
            batch_size=args.batch_size)
        if args.use_data_parallel:
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)

        train_loader.set_sample_list_generator(train_reader, places=place)
        valid_loader.set_sample_list_generator(valid_reader, places=place)

        save_parameters = (not args.use_data_parallel) or (
            args.use_data_parallel and
            fluid.dygraph.parallel.Env().local_rank == 0)
        best_top1 = 0
        for epoch in range(args.epochs):
            logger.info('Epoch {}, lr {:.6f}'.format(
                epoch, optimizer.current_step_lr()))
            train_top1, train_top5 = train(model, train_loader, optimizer,
                                           epoch, args)
            logger.info("Epoch {}, train_top1 {:.6f}, train_top5 {:.6f}".
                        format(epoch, train_top1, train_top5))
            valid_top1, valid_top5 = valid(model, valid_loader, epoch, args)
            if valid_top1 > best_top1:
                best_top1 = valid_top1
                if save_parameters:
                    fluid.save_dygraph(model.state_dict(),
                                       args.model_save_dir + "/best_model")
            logger.info(
                "Epoch {}, valid_top1 {:.6f}, valid_top5 {:.6f}, best_valid_top1 {:6f}".
                format(epoch, valid_top1, valid_top5, best_top1))


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    main(args)
