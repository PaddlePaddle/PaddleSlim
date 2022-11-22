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
import paddle


class Architect(object):
    def __init__(self, model, eta, arch_learning_rate, unrolled, parallel):
        self.network_momentum = 0.9
        self.network_weight_decay = 3e-4
        self.eta = eta
        self.model = model
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=arch_learning_rate,
            beta1=0.5,
            beta2=0.999,
            weight_decay=paddle.regularizer.L2Decay(coeff=1e-3),
            parameters=self.model.arch_parameters())
        self.unrolled = unrolled
        self.parallel = parallel
        if self.unrolled:
            self.unrolled_model = self.model.new()
            self.unrolled_model_params = [
                p for p in self.unrolled_model.parameters()
                if p.name not in
                [a.name
                 for a in self.unrolled_model.arch_parameters()] and p.trainable
            ]
            self.unrolled_optimizer = paddle.optimizer.Momentum(
                self.eta,
                self.network_momentum,
                regularization=paddle.regularizer.L2Decay(
                    coeff=self.network_weight_decay),
                parameter_list=self.unrolled_model_params)

        if self.parallel:
            strategy = paddle.fluid.dygraph.parallel.prepare_context()
            self.parallel_model = paddle.DataParallel(self.model, strategy)
            if self.unrolled:
                self.parallel_unrolled_model = paddle.DataParallel(
                    self.unrolled_model, strategy)

    def get_model(self):
        return self.parallel_model if self.parallel else self.model

    def step(self, input_train, target_train, input_valid, target_valid):
        if self.unrolled:
            params_grads = self._backward_step_unrolled(
                input_train, target_train, input_valid, target_valid)
            self.optimizer.apply_gradients(params_grads)
        else:
            loss = self._backward_step(input_valid, target_valid)
            self.optimizer.minimize(loss)
        self.optimizer.clear_gradients()

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        if self.parallel:
            loss = self.parallel_model.scale_loss(loss)
            loss.backward()
            self.parallel_model.apply_collective_grads()
        else:
            loss.backward()
        return loss

    def _backward_step_unrolled(self, input_train, target_train, input_valid,
                                target_valid):
        self._compute_unrolled_model(input_train, target_train)
        unrolled_loss = self.unrolled_model._loss(input_valid, target_valid)

        if self.parallel:
            unrolled_loss = self.parallel_unrolled_model.scale_loss(
                unrolled_loss)
            unrolled_loss.backward()
            self.parallel_unrolled_model.apply_collective_grads()
        else:
            unrolled_loss.backward()

        vector = [
            paddle.to_tensor(data=param._grad_ivar().numpy())
            for param in self.unrolled_model_params
        ]
        arch_params_grads = [
            (alpha, paddle.to_tensor(data=ualpha._grad_ivar().numpy()))
            for alpha, ualpha in zip(self.model.arch_parameters(),
                                     self.unrolled_model.arch_parameters())
        ]
        self.unrolled_model.clear_gradients()

        implicit_grads = self._hessian_vector_product(vector, input_train,
                                                      target_train)
        for (p, g), ig in zip(arch_params_grads, implicit_grads):
            new_g = g - (ig * self.unrolled_optimizer.current_step_lr())
            paddle.assign(new_g.detach(), g)
        return arch_params_grads

    def _compute_unrolled_model(self, input, target):
        for x, y in zip(self.unrolled_model.parameters(),
                        self.model.parameters()):
            paddle.assign(y.detach(), x)

        loss = self.unrolled_model._loss(input, target)
        if self.parallel:
            loss = self.parallel_unrolled_model.scale_loss(loss)
            loss.backward()
            self.parallel_unrolled_model.apply_collective_grads()
        else:
            loss.backward()

        self.unrolled_optimizer.minimize(loss)
        self.unrolled_model.clear_gradients()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r * paddle.rsqrt(
            paddle.fluid.layers.sum(
                [paddle.sum(x=paddle.square(v)) for v in vector]))

        model_params = [
            p for p in self.model.parameters()
            if p.name not in [a.name for a in self.model.arch_parameters()] and
            p.trainable
        ]
        for param, grad in zip(model_params, vector):
            param_p = param + grad * R
            paddle.assign(param_p.detach(), param)
        loss = self.model._loss(input, target)
        if self.parallel:
            loss = self.parallel_model.scale_loss(loss)
            loss.backward()
            self.parallel_model.apply_collective_grads()
        else:
            loss.backward()

        grads_p = [
            paddle.to_tensor(data=param._grad_ivar().numpy())
            for param in self.model.arch_parameters()
        ]

        for param, grad in zip(model_params, vector):
            param_n = param - grad * R * 2
            paddle.assign(param_n.detach(), param)
        self.model.clear_gradients()

        loss = self.model._loss(input, target)
        if self.parallel:
            loss = self.parallel_model.scale_loss(loss)
            loss.backward()
            self.parallel_model.apply_collective_grads()
        else:
            loss.backward()

        grads_n = [
            paddle.to_tensor(data=param._grad_ivar().numpy())
            for param in self.model.arch_parameters()
        ]
        for param, grad in zip(model_params, vector):
            param_o = param + grad * R
            paddle.assign(param_o.detach(), param)
        self.model.clear_gradients()
        arch_grad = [(p - n) / (2 * R) for p, n in zip(grads_p, grads_n)]
        return arch_grad
