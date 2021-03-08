#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle

lr_strategy = 'cosine_decay'
l2_decay = 1e-4
step_epochs = [30, 60, 90]
momentum_rate = 0.9
warm_up_epochs = 5.0
num_epochs = 120
decay_epochs = 2.4
decay_rate = 0.97
total_images = 1281167


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

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.lr_strategy = lr_strategy
        self.l2_decay = l2_decay
        self.momentum_rate = momentum_rate
        self.step_epochs = step_epochs
        self.num_epochs = num_epochs
        self.warm_up_epochs = warm_up_epochs
        self.decay_epochs = decay_epochs
        self.decay_rate = decay_rate
        self.total_images = total_images
        if args.use_gpu:
            devices_num = paddle.fluid.core.get_cuda_device_count()
        else:
            devices_num = int(os.environ.get('CPU_NUM', 1))

        self.step = int(
            math.ceil(float(self.total_images) / self.batch_size) / devices_num)

    def cosine_decay(self):
        """cosine decay with Momentum optimizer

        Returns:
            a cosine_decay optimizer
        """
        learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=self.lr,
            T_max=self.step * self.num_epochs,
            verbose=False)
        optimizer = paddle.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=self.momentum_rate,
            weight_decay=paddle.regularizer.L2Decay(self.l2_decay))
        return optimizer

    def piecewise_decay(args):
        bd = [step * e for e in args.step_epochs]
        lr = [args.lr * (0.1**i) for i in range(len(bd) + 1)]
        learning_rate = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=bd, values=lr, verbose=False)
        optimizer = paddle.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=args.momentum_rate,
            weight_decay=paddle.regularizer.L2Decay(args.l2_decay))
        return optimizer


def create_optimizer(args):
    Opt = Optimizer(args)
    optimizer = getattr(Opt, lr_strategy)()

    return optimizer
