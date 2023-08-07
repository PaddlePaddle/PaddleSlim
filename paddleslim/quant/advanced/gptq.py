# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License.

import math
import time
import numpy as np
import paddle
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import ColumnParallelLinear, RowParallelLinear

from .utils import compute_scales
__all__ = ['GPTQ']


class GPTQ(nn.Layer):
    def __init__(self,
                 layer,
                 quant_bits=4,
                 weight_quant_method='abs_max_channel_wise'):
        '''
        The implementation of GPTQ(https://arxiv.org/abs/2210.17323). 
        The codes here are based on https://github.com/IST-DASLab/gptq.
        
        Args:
        layer (paddle.nn.Layer): Layer object.
        quant_bits (int, optional): Number of bits to quantize the weight. Default: 4.
        weight_quant_method (str, optional): Method of weight quantization. Choosen from abs_max, abs_max_channel_wise and avg. Default: abs_max_channel_wise.

        Examples:
        .. code-block:: python
        
        from paddleslim.quant.advanced import GPTQ
        for cur_name, cur_layer in model.named_sublayers():
            if type(cur_layer) == paddle.nn.Linear:
                gptq_layer = GPTQ(cur_layer)
                # sample data
                for data in dataloader():
                    model(data)
                # quant weight
                gptq_layer.fasterquant()  
        '''
        super(GPTQ, self).__init__()
        self.layer = layer
        assert hasattr(layer,
                       'weight'), "Layer {} has no attribute 'weight'".format(
                           layer.full_name())
        assert type(self.layer) in [
            nn.Linear, ColumnParallelLinear, RowParallelLinear
        ], "Currently, GPTQ only supports linear layer and ColumnParallelLinear/RowParallelLinear layer"

        weight = layer.weight.t()

        self.rows = weight.shape[0]
        self.columns = weight.shape[1]
        self.hessian = paddle.zeros(
            (self.columns, self.columns), dtype='float32')
        self.nsamples = 0
        self.quantized = False
        self.weight_quant_method = weight_quant_method
        self.quant_bits = (1 << (quant_bits - 1)) - 1

    def forward(self, input):
        if not self.quantized:
            inp = input[0] if type(input) == tuple else input
            inp = inp.cast('float32')
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            tmp = inp.shape[0]
            if type(self.layer) in [
                    nn.Linear, ColumnParallelLinear, RowParallelLinear
            ]:
                if len(inp.shape) == 3:
                    inp = inp.reshape((-1, inp.shape[-1]))
                inp = inp.t()

            self.hessian *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp

            inp = math.sqrt(2 / self.nsamples) * inp
            self.hessian += paddle.matmul(inp, inp.t())
            del inp
        return self.layer(input)

    def fasterquant(self,
                    blocksize=128,
                    percdamp=.01,
                    groupsize=-1,
                    actorder=True):
        print('quant', self.layer.full_name())
        W = self.layer.weight.t().cast('float32')
        weight_scale = compute_scales(W.t(), method=self.weight_quant_method)
        weight_scale /= self.quant_bits
        tick = time.time()

        H = self.hessian
        del self.hessian
        dead = paddle.where(paddle.diag(H) == 0)
        H[dead, dead] = 1
        W[:, dead] = 0
        del dead
        if actorder:
            perm = paddle.argsort(paddle.diag(H), descending=True)
            W = W.transpose((1, 0))
            W = W[perm].transpose((1, 0))
            H = H[perm].transpose((1, 0))
            H = H[perm].transpose((1, 0))

        Losses = paddle.zeros_like(W)
        Q = paddle.zeros_like(W)

        damp = percdamp * paddle.mean(paddle.diag(H))
        diag = paddle.arange(self.columns)
        H[diag, diag] += damp

        H = paddle.inverse(H)
        H = paddle.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2]
            Q1 = paddle.zeros_like(W1)
            Err1 = paddle.zeros_like(W1)
            Losses1 = paddle.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        weight_scale = compute_scales(
                            W[:, (i1 + i):(i1 + i + groupsize)].t(),
                            method=self.weight_quant_method)
                        weight_scale /= self.quant_bits

                q = paddle.clip(
                    paddle.round(w / weight_scale), -self.quant_bits - 1,
                    self.quant_bits)
                q = q * weight_scale
                Q1[:, i] = q
                Losses1[:, i] = (w - q)**2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
                del w, d, q, err1
                paddle.device.cuda.empty_cache()

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            del Q1, Losses1
            if Hinv[i1:i2, i2:].shape[1] != 0:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            del Err1, W1, Hinv1
            paddle.device.cuda.empty_cache()

        print('time %.2f' % (time.time() - tick))
        print('error', paddle.sum(Losses).item())

        if actorder:
            invperm = paddle.argsort(perm)
            Q = Q.transpose((1, 0))
            Q = Q[invperm].transpose((1, 0))
            del invperm, perm

        param = self.layer.weight
        Q = Q.t().cast(self.layer.weight.dtype)
        paddle.assign(Q, output=param)

        self.quantized = True
        del H, Q, Hinv, W, Losses
        paddle.device.cuda.empty_cache()
