#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle.fluid as fluid
import paddle.fluid.layers.ops as ops
from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay


class CosineDecayWarmup(LearningRateDecay):
    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 warmup_epochs,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(CosineDecayWarmup, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.step_each_epoch = step_each_epoch
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs

    def step(self):
        if (self.step_num / self.step_each_epoch) < self.warmup_epochs:
            decayed_lr = self.learning_rate * (self.step_num / (
                self.step_each_epoch * self.warmup_epochs))
        else:
            decayed_lr = self.learning_rate * 0.5 * (fluid.layers.cos(
                self.create_lr_var((self.step_num - self.warmup_epochs *
                                    self.step_each_epoch) * math.pi /
                                   (self.epochs * self.step_each_epoch))) + 1)
        return decayed_lr


class Optimizer(object):
    """A class used to represent several optimizer methods
    Attributes:
        batch_size: batch size on all devices.
        lr: learning rate.
        lr_strategy: learning rate decay strategy.
        l2_decay: l2_decay parameter.
        momentum_rate: momentum rate when using Momentum optimizer.
        step_epochs: piecewise decay steps.
        num_epochs: number of total epochs.
        total_images: total images.
        step: total steps in the an epoch.
        
    """

    def __init__(self, args, parameter_list):
        self.parameter_list = parameter_list
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.lr_strategy = args.lr_strategy
        self.l2_decay = args.l2_decay
        self.momentum_rate = args.momentum_rate
        self.step_epochs = args.step_epochs
        self.num_epochs = args.num_epochs
        self.warm_up_epochs = args.warm_up_epochs
        self.decay_epochs = args.decay_epochs
        self.decay_rate = args.decay_rate
        self.total_images = args.total_images

        self.step = int(math.ceil(float(self.total_images) / self.batch_size))

    def piecewise_decay(self):
        """piecewise decay with Momentum optimizer
            Returns:
            a piecewise_decay optimizer
        """
        bd = [self.step * e for e in self.step_epochs]
        lr = [self.lr * (0.1**i) for i in range(len(bd) + 1)]
        learning_rate = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=self.momentum_rate,
            regularization=fluid.regularizer.L2Decay(self.l2_decay),
            parameter_list=self.parameter_list)
        return optimizer

    def cosine_decay(self):
        """cosine decay with Momentum optimizer
        Returns:
            a cosine_decay optimizer
        """

        learning_rate = fluid.layers.cosine_decay(
            learning_rate=self.lr,
            step_each_epoch=self.step,
            epochs=self.num_epochs)
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=self.momentum_rate,
            regularization=fluid.regularizer.L2Decay(self.l2_decay),
            parameter_list=self.parameter_list)
        return optimizer

    def cosine_decay_warmup(self):
        """cosine decay with warmup
        Returns:
            a cosine_decay_with_warmup optimizer
        """

        learning_rate = CosineDecayWarmup(
            learning_rate=self.lr,
            step_each_epoch=self.step,
            epochs=self.num_epochs,
            warmup_epochs=5)
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=self.momentum_rate,
            regularization=fluid.regularizer.L2Decay(self.l2_decay),
            parameter_list=self.parameter_list)
        return optimizer

    def linear_decay(self):
        """linear decay with Momentum optimizer
        Returns:
            a linear_decay optimizer
        """

        end_lr = 0
        learning_rate = fluid.layers.polynomial_decay(
            self.lr, self.step, end_lr, power=1)
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=self.momentum_rate,
            regularization=fluid.regularizer.L2Decay(self.l2_decay),
            parameter_list=self.parameter_list)

        return optimizer

    def adam_decay(self):
        """Adam optimizer
        Returns: 
            an adam_decay optimizer
        """

        return fluid.optimizer.Adam(
            learning_rate=self.lr, parameter_list=self.parameter_list)

    def cosine_decay_RMSProp(self):
        """cosine decay with RMSProp optimizer
        Returns: 
            an cosine_decay_RMSProp optimizer
        """

        learning_rate = fluid.layers.cosine_decay(
            learning_rate=self.lr,
            step_each_epoch=self.step,
            epochs=self.num_epochs)
        optimizer = fluid.optimizer.RMSProp(
            learning_rate=learning_rate,
            momentum=self.momentum_rate,
            regularization=fluid.regularizer.L2Decay(self.l2_decay),
            # Apply epsilon=1 on ImageNet dataset.
            epsilon=1,
            parameter_list=self.parameter_list)
        return optimizer

    def default_decay(self):
        """default decay
        Returns:
            default decay optimizer
        """

        optimizer = fluid.optimizer.Momentum(
            learning_rate=self.lr,
            momentum=self.momentum_rate,
            regularization=fluid.regularizer.L2Decay(self.l2_decay),
            parameter_list=self.parameter_list)
        return optimizer


def create_optimizer(args, parameter_list):
    Opt = Optimizer(args, parameter_list)
    optimizer = getattr(Opt, args.lr_strategy)()

    return optimizer
