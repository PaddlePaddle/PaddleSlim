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
import paddle
import numpy as np
from .utils import compute_scales
from .metrics import mse_loss
__all__ = ['AWQSearch']

class AWQSearch():
    def __init__(self,
                 n_grid=20,
                 bits_length=4,
                 weight_quant_method='groupwise',
                 group_size=128,
                 loss_function=mse_loss):
        '''
        The implementation of AutoScale from AWQ(https://arxiv.org/pdf/2306.00978.pdf).
        '''
        self.n_grid = n_grid
        self.bits_length = bits_length
        self.weight_quant_method = weight_quant_method
        self.bnt = (1 << (bits_length - 1)) - 1
        self.group_size = group_size
        self.loss_function = loss_function
        
    def search(self, layer_name, sampled_input, act_abs_max, weight):
        act = sampled_input
        act.stop_gradient = True
        print('[awq search] search input of %s' % layer_name)
        dtype = weight.dtype
        origin_out = paddle.matmul(act, weight)
        best_error = float('inf')
        best_ratio = -1
        best_scales = None
        
        for ratio in range(self.n_grid):
            ratio = ratio * 1 / self.n_grid
            act_abs_max_tmp = act_abs_max.detach().clone().cast('float32')
            scales = paddle.clip(paddle.pow(act_abs_max_tmp, ratio), min=1e-4)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales = scales.cast(dtype)
            new_weight = weight * scales.reshape([-1, 1])
            new_act = act / scales
            quant_scale = compute_scales(
                new_weight, method=self.weight_quant_method, group_size=self.group_size)
            if self.weight_quant_method == 'groupwise':
                quant_scale = paddle.repeat_interleave(quant_scale.cast('float32'), self.group_size, 0).cast(dtype)
            quant_weight = paddle.clip(
                paddle.round(new_weight / quant_scale * self.bnt),
                -self.bnt - 1, self.bnt)
            quant_dequant_weight = quant_weight / self.bnt * quant_scale
            new_out = paddle.matmul(new_act,
                                    quant_dequant_weight)
            loss = self.loss_function(origin_out, new_out).numpy()
            is_best = loss < best_error
            if is_best:
                print('find better ratio: {}, loss: {}'.format(ratio, loss))
                best_error = loss
                best_ratio = ratio
                best_scales = scales
        
        if best_scales is None:
            best_scales = paddle.ones(scales.shape, dtype=dtype)
            print('Cannot find better ratio.')
        else:
            print('Best ratio :{}, minimal loss : {}.'.format(best_ratio, best_error))
        return best_scales
