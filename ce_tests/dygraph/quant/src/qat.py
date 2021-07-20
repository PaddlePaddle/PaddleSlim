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

from __future__ import division
from __future__ import print_function

import argparse
import contextlib
import os

import time
import math
import numpy as np

import paddle
from paddle.distributed import ParallelEnv

from paddle.optimizer.lr import PiecewiseDecay
from paddle.metric.metrics import Accuracy
import paddle.vision.models as models

from paddleslim import QAT
from paddle.fluid.contrib.slim.quantization import ImperativeQuantAware

from imagenet_dataset import ImageNetDataset


def make_optimizer(step_per_epoch, parameter_list=None):
    assert FLAGS.lr_scheduler == 'piecewise'

    base_lr = FLAGS.lr
    lr_scheduler = FLAGS.lr_scheduler
    momentum = FLAGS.momentum
    weight_decay = FLAGS.weight_decay
    milestones = FLAGS.milestones

    boundaries = [step_per_epoch * e for e in milestones]
    values = [base_lr * (0.1**i) for i in range(len(boundaries) + 1)]
    learning_rate = PiecewiseDecay(boundaries=boundaries, values=values)

    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        parameters=parameter_list)

    return optimizer


def main():
    # create model
    model_list = [x for x in models.__dict__["__all__"]]
    assert FLAGS.arch in model_list, \
        "Expected FLAGS.arch in {}, but received {}".format(
        model_list, FLAGS.arch)
    model = models.__dict__[FLAGS.arch](pretrained=not FLAGS.resume)

    # quantize model
    if FLAGS.enable_quant:
        if not FLAGS.use_naive_api:
            print("use slim api")
            quant_config = {
                'weight_quantize_type': FLAGS.weight_quantize_type,
            }
            dygraph_qat = QAT(quant_config)
        else:
            print("use navie api")
            dygraph_qat = ImperativeQuantAware(
                weight_quantize_type=FLAGS.weight_quantize_type, )
        dygraph_qat.quantize(model)

    # prepare
    model = paddle.Model(model)
    if FLAGS.resume is not None:
        print("Resume from " + FLAGS.resume)
        model.load(FLAGS.resume)

    train_dataset = ImageNetDataset(
        os.path.join(FLAGS.data, 'train'), mode='train')
    val_dataset = ImageNetDataset(
        os.path.join(FLAGS.data, FLAGS.val_dir), mode='val')

    optim = make_optimizer(
        np.ceil(
            float(len(train_dataset)) / FLAGS.batch_size /
            ParallelEnv().nranks),
        parameter_list=model.parameters())

    model.prepare(optim, paddle.nn.CrossEntropyLoss(), Accuracy(topk=(1, 5)))

    # test
    if FLAGS.eval_only:
        model.evaluate(
            val_dataset,
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.num_workers)
        return

    # train
    output_dir = os.path.join(FLAGS.output_dir, "checkpoint",
                              FLAGS.arch + "_checkpoint",
                              time.strftime('%Y-%m-%d-%H-%M', time.localtime()))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.fit(train_dataset,
              val_dataset,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epoch,
              save_dir=output_dir,
              num_workers=FLAGS.num_workers)

    # save
    if FLAGS.enable_quant:
        quant_output_dir = os.path.join(FLAGS.output_dir, FLAGS.arch, "model")
        input_spec = paddle.static.InputSpec(
            shape=[None, 3, 224, 224], dtype='float32')
        dygraph_qat.save_quantized_model(model.network, quant_output_dir,
                                         [input_spec])
        print("save all checkpoints in " + output_dir)
        print("save quantized inference model in " + quant_output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training on ImageNet")

    # model
    parser.add_argument(
        "--arch", type=str, default='mobilenet_v2', help="model name")
    parser.add_argument(
        "--resume", default=None, type=str, help="checkpoint path to resume")
    parser.add_argument(
        "--eval_only", action='store_true', help="only evaluate the model")
    parser.add_argument(
        "--output_dir", type=str, default='output', help="save dir")

    # data
    parser.add_argument(
        '--data',
        metavar='DIR',
        default="",
        help='path to dataset '
        '(should have subdirectories named "train" and "val"')
    parser.add_argument(
        '--val_dir',
        default="val_hapi",
        help='the dir that saves val images for paddle.Model')

    # train
    parser.add_argument(
        "-e", "--epoch", default=1, type=int, help="number of epoch")
    parser.add_argument(
        "-b", "--batch_size", default=10, type=int, help="batch size")
    parser.add_argument(
        "-n", "--num_workers", default=2, type=int, help="dataloader workers")
    parser.add_argument(
        '--lr',
        default=0.0001,
        type=float,
        metavar='LR',
        help='initial learning rate')
    parser.add_argument(
        "--lr-scheduler",
        default='piecewise',
        type=str,
        help="learning rate scheduler")
    parser.add_argument(
        "--milestones",
        nargs='+',
        type=int,
        default=[1, 2, 3, 4, 5],
        help="piecewise decay milestones")
    parser.add_argument(
        "--weight-decay", default=1e-4, type=float, help="weight decay")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")

    # quant
    parser.add_argument(
        "--enable_quant", action='store_true', help="enable quant model")
    parser.add_argument(
        "--use_naive_api", action='store_true', help="use the navie api")
    parser.add_argument(
        "--weight_quantize_type", type=str, default='abs_max', help="")

    FLAGS = parser.parse_args()
    assert FLAGS.data, "error: must provide data path"

    main()
