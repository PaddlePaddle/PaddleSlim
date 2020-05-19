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
import paddle.fluid as fluid
import paddle.nn.functional as F


class DML(fluid.dygraph.Layer):
    def __init__(self, model, use_parallel):
        super(DML, self).__init__()
        self.model = model
        self.use_parallel = use_parallel
        self.model_num = len(self.model)
        if self.use_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()
            self.model = [
                fluid.dygraph.parallel.DataParallel(m, strategy)
                for m in self.model
            ]

    def full_name(self):
        return [m.full_name() for m in self.model]

    def forward(self, input):
        outputs = [m(input) for m in self.model]
        return outputs

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
                fluid.layers.mean(
                    fluid.layers.cross_entropy(logits[i], labels)))
        return ce_losses

    def kl_loss(self, logits):
        assert len(
            logits
        ) == self.model_num, "The number of logits must match the number of models"
        kl_losses = []
        for i in range(self.model_num):
            cur_model_kl_loss = 0
            for j in range(len(logits)):
                if i != j:
                    x = F.log_softmax(logits[i], axis=1)
                    y = fluid.layers.softmax(logits[j], axis=1)
                    cur_model_kl_loss += fluid.layers.kldiv_loss(
                        x, y, reduction='batchmean')
            kl_losses.append(cur_model_kl_loss / (len(logits) - 1))
        return kl_losses

    def loss(self, logits, labels):
        gt_losses = self.ce_loss(logits, labels)
        kl_losses = self.kl_loss(logits)
        return [a + b for a, b in zip(gt_losses, kl_losses)]

    def acc(self, logits, labels, k):
        accs = [
            fluid.layers.accuracy(
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
