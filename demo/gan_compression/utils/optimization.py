#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import paddle
import paddle.nn as nn


class Optimizer:
    def __init__(self,
                 lr,
                 scheduler,
                 step_per_epoch,
                 nepochs,
                 nepochs_decay,
                 args,
                 parameter_list=None):
        self.lr = lr
        self.scheduler = scheduler
        self.step_per_epoch = step_per_epoch
        self.nepochs = nepochs
        self.nepochs_decay = nepochs_decay
        self.args = args
        self.parameter_list = parameter_list
        self.optimizer = self.lr_scheduler()

    ### NOTE(ceci3): add more scheduler
    def lr_scheduler(self):
        if self.scheduler == 'linear':

            def decay_relu(epoch):
                lr_l = 1.0 - max(0, epoch + 1 -
                                 self.nepochs) / float(self.nepochs_decay + 1)
                return lr_l

            self.scheduler_lr = paddle.optimizer.lr.LambdaDecay(
                self.lr, lr_lambda=decay_relu, last_epoch=self.nepochs)
        elif self.scheduler == 'step':
            pass
        elif self.scheduler == 'cosine':
            pass
        else:
            return NotImplementedError(
                'learning rate policy [%s] is not implemented', opt.lr_policy)

        optimizer = paddle.optimizer.Adam(
            learning_rate=self.scheduler_lr,
            beta1=self.args.beta1,
            beta2=0.999,
            parameters=self.parameter_list)
        return optimizer
