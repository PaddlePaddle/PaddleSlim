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
import weakref
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.optim import Optimizer


class TORCH_DML(nn.Module):
    def __init__(self, model):
        super(TORCH_DML, self).__init__()
        self.model = model
        self.parnters = [model]
        device = next(self.model.parameters()).device

        self.parnters.append(type(self.model)(num_classes=10).to(device))

#        self.parnters.append(copy.deepcopy(self.model).to(device))

    def forward(self, input):
        if self.model.training:
            return [m(input) for m in self.parnters]
        else:
            return self.model(input)

    def opt(self, optimizer):
        optimizers = []
        for parnter in self.parnters:
            new_opt = copy.deepcopy(optimizer)
            new_opt.param_groups = []
            new_opt.add_param_group({"params": parnter.parameters()})
            optimizers.append(new_opt)
        self.optimizer = DMLOptimizer(optimizers, self)
        return self.optimizer

    def _clone_scheduler(self, scheduler, optimizer):
        scheduler = copy.deepcopy(scheduler)
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        scheduler.optimizer = optimizer
        scheduler.last_epoch -= 1
        # Initialize epoch and base learning rates
        if scheduler.last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".
                        format(i))
        scheduler.base_lrs = list(
            map(lambda group: group['initial_lr'], optimizer.param_groups))

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        scheduler.optimizer.step = with_counter(scheduler.optimizer.step)
        scheduler.optimizer._step_count = 0
        scheduler._step_count = 0

        scheduler.step()
        return scheduler

    def lr(self, scheduler):
        schedulers = []
        for opt in self.optimizer.optimizers:
            new_scheduler = self._clone_scheduler(scheduler, opt)
            schedulers.append(new_scheduler)
        self.scheduler = DMLScheduler(schedulers, self)
        return self.scheduler

    def nll_loss(self, logit, label):
        logit = F.softmax(logit, dim=1)
        return F.cross_entropy(logit, label)

    def kl_loss(self, logit0, logit1):
        logit0 = F.log_softmax(logit0, dim=1)
        logit1 = F.softmax(logit1, dim=1)
        return F.kl_div(logit0, logit1, reduction='batchmean')

    def dml_loss(self, logits, label, gt_loss_func=None, dist_loss_func=None):
        gt_loss_func = gt_loss_func if gt_loss_func is not None else self.nll_loss
        dist_loss_func = dist_loss_func if dist_loss_func is not None else self.kl_loss

        self.losses = []
        for i in range(len(logits)):
            logit = logits[i]
            cur_loss = gt_loss_func(logit, label)
            for j in range(len(logits)):
                if i != j:
                    dist_loss = dist_loss_func(logits[i], logits[j])
                    cur_loss += dist_loss
            self.losses.append(cur_loss)
        return self.losses[0]

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


class DMLOptimizer(object):
    def __init__(self, optimizers, model):
        self.optimizers = optimizers
        self.model = model

    def step(self):
        for loss, optimizer in zip(self.model.losses, self.optimizers):
            loss.backward(retain_graph=True)
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()


class DMLScheduler(object):
    def __init__(self, schedulers, model):
        self.schedulers = schedulers
        self.model = model

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()
