# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import math
import paddle
from paddle.autograd import PyLayer


def round(x):
    sign = paddle.sign(x)
    x = sign * paddle.floor(paddle.abs(x) + 0.5)
    return x


class LsqFunc(PyLayer):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, per_channel=False, quant_axis=0):
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp, per_channel, quant_axis
        if per_channel:
            sizes = weight.shape
            weight = weight.reshape((weight.shape[quant_axis], -1))
            weight = weight.transpose((1, 0))
            alpha = paddle.broadcast_to(alpha, weight.shape)
            quant_w = round(paddle.divide(weight, alpha)).clip(Qn, Qp)
            quant_w = quant_w * alpha
            quant_w = quant_w.transpose((1, 0))
            quant_w = quant_w.reshape(sizes)
        else:
            quant_w = round(paddle.divide(weight, alpha)).clip(Qn, Qp)
            quant_w = quant_w * alpha
        return quant_w

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensor()
        g, Qn, Qp, per_channel, quant_axis = ctx.other
        if per_channel:
            sizes = weight.shape
            weight = weight.reshape((weight.shape[quant_axis], -1))
            weight = weight.transpose((1, 0))
            alpha = paddle.broadcast_to(alpha, weight.shape)
            q_w = paddle.divide(weight, alpha)
            q_w = q_w.transpose((1, 0))
            q_w = q_w.reshape(sizes)
        else:
            q_w = paddle.divide(weight, alpha)
        lower_flag = paddle.cast((q_w < Qn), 'float32')
        upper_flag = paddle.cast((q_w > Qp), 'float32')
        middle_flag = 1.0 - lower_flag - upper_flag
        if per_channel:
            grad_alpha = (
                (lower_flag * Qn + upper_flag * Qp + middle_flag * round(q_w) -
                 middle_flag * q_w) * grad_weight * g)
            grad_alpha = grad_alpha.reshape((grad_alpha.shape[quant_axis],
                                             -1)).sum(axis=1)
        else:
            grad_alpha = ((
                (lower_flag * Qn + upper_flag * Qp + middle_flag * round(q_w)
                 - middle_flag * q_w) * grad_weight * g).sum().unsqueeze(
                     axis=0)[0])
        grad_weight = middle_flag * grad_weight
        return grad_weight, grad_alpha


class LsqPlusActFunc(PyLayer):
    @staticmethod
    def forward(ctx, x, alpha, beta, g, Qn, Qp):
        ctx.save_for_backward(x, alpha, beta)
        ctx.other = g, Qn, Qp
        quant_x = round(paddle.divide((x - beta), alpha)).clip(Qn, Qp)
        return quant_x * alpha + beta

    @staticmethod
    def backward(ctx, grad_x):
        x, alpha, beta = ctx.saved_tensor()
        g, Qn, Qp = ctx.other
        q_x = (x - beta) / alpha
        lower_flag = paddle.cast((q_x < Qn), 'float32')
        upper_flag = paddle.cast((q_x > Qp), 'float32')
        middle_flag = 1.0 - lower_flag - upper_flag
        grad_alpha = ((
            (lower_flag * Qn + upper_flag * Qp + middle_flag * round(q_x) -
             middle_flag * q_x) * grad_x * g).sum().unsqueeze(axis=0)[0])
        grad_beta = (((lower_flag + upper_flag) * grad_x * g).sum().unsqueeze(
            axis=0)[0])
        grad_x = middle_flag * grad_x
        return grad_x, grad_alpha, grad_beta
