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

import copy
import paddle


class DML(paddle.nn.Layer):
    def __init__(self, model, use_parallel=False):
        super(DML, self).__init__()
        self.model = model
        self.use_parallel = use_parallel
        self.model_num = len(self.model)

    def full_name(self):
        return [m.full_name() for m in self.model]

    def forward(self, input):
        return [m(input) for m in self.model]

    def opt(self, optimizer):
        assert len(
            optimizer
        ) == self.model_num, "The number of optimizers must match the number of models"
        optimizer = DMLOptimizers(self.model, optimizer, self.use_parallel)
        return optimizer

    def ce_loss(self, logits, labels):
        assert len(
            logits
        ) == self.model_num, "The number of logits must match the number of models"
        ce_losses = []
        for i in range(self.model_num):
            ce_losses.append(
                paddle.mean(
                    paddle.nn.functional.softmax_with_cross_entropy(logits[i],
                                                                    labels)))
        return ce_losses

    def kl_loss(self, logits):
        assert len(
            logits
        ) == self.model_num, "The number of logits must match the number of models"
        if self.model_num == 1:
            return []
        kl_losses = []
        for i in range(self.model_num):
            cur_kl_loss = 0
            for j in range(self.model_num):
                if i != j:
                    log_softmax = paddle.nn.LogSoftmax(axis=1)
                    x = log_softmax(logits[i])

                    y = paddle.nn.functional.softmax(logits[j], axis=1)
                    cur_kl_loss += paddle.nn.functional.kl_div(
                        x, y, reduction='batchmean')
            kl_losses.append(cur_kl_loss / (self.model_num - 1))
        return kl_losses

    def loss(self, logits, labels):
        gt_losses = self.ce_loss(logits, labels)
        kl_losses = self.kl_loss(logits)
        if self.model_num > 1:
            return [a + b for a, b in zip(gt_losses, kl_losses)]
        else:
            return gt_losses

    def acc(self, logits, labels, k):
        accs = [
            paddle.metric.accuracy(
                input=l, label=labels, k=k) for l in logits
        ]
        return accs

    def train(self):
        for m in self.model:
            m.train()

    def eval(self):
        for m in self.model:
            m.eval()


class DMLOptimizers(object):
    def __init__(self, model, optimizer, use_parallel):
        self.model = model
        self.optimizer = optimizer
        self.use_parallel = use_parallel

    def minimize(self, losses):
        assert len(losses) == len(
            self.optimizer
        ), "The number of losses must match the number of optimizers"
        for i in range(len(losses)):
            if self.use_parallel:
                losses[i] = self.model[i].scale_loss(losses[i])
                losses[i].backward()
                self.model[i].apply_collective_grads()
            else:
                losses[i].backward()
            self.optimizer[i].minimize(losses[i])
            self.model[i].clear_gradients()

    def get_lr(self):
        current_step_lr = [opt.current_step_lr() for opt in self.optimizer]
        return current_step_lr
