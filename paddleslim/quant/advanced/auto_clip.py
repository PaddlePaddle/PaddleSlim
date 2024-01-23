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
"""
AutoClip.
"""
import paddle
import paddle.nn as nn
import numpy as np
from .utils import fake_quant
from .metrics import mse_loss
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear, )
__all__ = ['AutoClip']


class AutoClip(nn.Layer):
    """
    AutoClip from AWQ[https://arxiv.org/abs/2306.00978]
    """

    def __init__(
            self,
            model,
            weight_bits=4,
            weight_quant_method='groupwise',
            loss_function=mse_loss,
            sample_function=None,
            n_grid=20,
            max_shrink=0.5,
            n_sample_token=512,
            group_size=128, ):
        super(AutoClip, self).__init__()
        self.model = model
        self.weight_bits = weight_bits
        self.weight_method = weight_quant_method
        self.loss_function = loss_function
        self.n_grid = n_grid
        self.max_shrink = max_shrink
        self.n_sample_token = n_sample_token
        self.bnt = (1 << (self.weight_bits - 1)) - 1
        self.sampled_inputs = {}
        self.sample_function = sample_function
        self.group_size = group_size

        self._apply_hook()

    def _apply_hook(self):
        self._forward_hook_list = []
        for _, sub_layer in self.model.named_sublayers():
            if type(sub_layer) in [
                    ColumnParallelLinear, RowParallelLinear, paddle.nn.Linear
            ]:
                forward_pre_hook_handle = sub_layer.register_forward_pre_hook(
                    self._forward_pre_hook)
                self._forward_hook_list.append(forward_pre_hook_handle)

    def _forward_pre_hook(self, layer, input):
        self._sample_scale(input, layer.full_name())
        return input

    def _sample_scale(self, input, name):
        input = input[0] if type(input) == tuple else input
        input.stop_gradient = True
        if name not in self.sampled_inputs:
            self.sampled_inputs[name] = input
        else:
            if self.sample_function is not None:
                self.sampled_inputs[name] = self.sample_function.sample(
                    input, self.sampled_inputs[name], name)
            else:
                self.sampled_inputs[name] = input

    def auto_clip(self, group_size=128, oc_batch_size=256):
        """
        search clip scale for each layer and update the layer's weight
        """
        for sub_name, sub_layer in self.model.named_sublayers():
            name = sub_layer.full_name()
            if name not in self.sampled_inputs or 'out_linear' in sub_name:
                continue

            weight = sub_layer.weight.cast('float16')
            weight_t = paddle.transpose(weight, perm=[1, 0])
            x = self.sampled_inputs[name].cast('float16')
            print('AutoClipping', sub_name, name, x.shape, weight.shape)
            x = x.reshape([-1, x.shape[-1]])
            x = x.reshape([1, x.shape[0], -1, group_size])
            x = x[:, 0::x.shape[1] // self.n_sample_token]
            weight_t = weight_t.reshape([weight_t.shape[0], 1, -1, group_size])
            oc_batch_size = oc_batch_size if weight_t.shape[
                0] % oc_batch_size == 0 else 128  # prevent OOM
            assert weight_t.shape[0] % oc_batch_size == 0

            w_all = weight_t
            best_max_val_all = []

            for i_b in range(weight_t.shape[0] // oc_batch_size):
                w = w_all[i_b * oc_batch_size:(i_b + 1) * oc_batch_size]

                org_max_val = w.abs().max(
                    axis=-1, keepdim=True)  # co, 1, n_group, 1
                best_max_val = org_max_val.clone()
                min_errs = paddle.ones_like(org_max_val, dtype='float16') * 1e9
                org_out = (x * w).sum(axis=-1)  # co, n_token, n_group
                for i_s in range(int(self.max_shrink * self.n_grid)):
                    max_val = org_max_val * (1 - i_s / self.n_grid)
                    max_val_tmp = max_val
                    cur_w = paddle.where(w > max_val_tmp, max_val_tmp, w)
                    cur_w = paddle.where(cur_w < -max_val_tmp, -max_val_tmp,
                                         cur_w)
                    org_w_shape = cur_w.shape
                    cur_w_r = cur_w.reshape([-1,
                                             self.group_size]).transpose([1, 0])
                    quant_dequant_weight = fake_quant(
                        cur_w_r, method='abs_max_channel_wise', weight_bits=4)
                    quant_dequant_weight = quant_dequant_weight.transpose(
                        [1, 0]).reshape(org_w_shape)
                    cur_out = (x * quant_dequant_weight).sum(axis=-1)
                    # co, 1, n_group, 1
                    tmp = (cur_out - org_out).detach().clone()
                    err = paddle.pow(tmp,
                                     2).mean(axis=1).reshape(min_errs.shape)
                    print('block {} search s {} err {}'.format(
                        i_b, i_s, err.mean().item()))
                    del cur_w, cur_out, quant_dequant_weight, tmp, cur_w_r
                    paddle.device.cuda.empty_cache()

                    cur_best_idx = paddle.where(err < min_errs)
                    if cur_best_idx[0].shape[0] != 0:
                        min_errs[cur_best_idx] = err[cur_best_idx]
                        best_max_val[cur_best_idx] = max_val[cur_best_idx]
                best_max_val_all.append(best_max_val)

                del org_out, org_max_val, min_errs, best_max_val, err, cur_best_idx, max_val_tmp, max_val, w
                paddle.device.cuda.empty_cache()

            best_max_val = paddle.concat(best_max_val_all, axis=0)
            best_max_val = paddle.squeeze(best_max_val, axis=1)
            for param in sub_layer.parameters(include_sublayers=False):
                if 'w_0' in param.name:
                    param_tmp = param.transpose(perm=[1, 0]).cast('float16')
                    tmp_shape = param_tmp.shape
                    param_tmp = param_tmp.reshape(
                        [best_max_val.shape[0], best_max_val.shape[1], -1])
                    best_max_val = paddle.tile(
                        best_max_val, repeat_times=(1, 1, param_tmp.shape[-1]))
                    param_tmp = paddle.where(param_tmp > best_max_val,
                                             best_max_val, param_tmp)
                    param_tmp = paddle.where(param_tmp < -best_max_val,
                                             -best_max_val, param_tmp)
                    param_tmp = param_tmp.reshape(tmp_shape).cast(param.dtype)
                    param_tmp = param_tmp.transpose(perm=[1, 0])
                    paddle.assign(param_tmp, output=param)
                    del param_tmp
                    paddle.device.cuda.empty_cache()
                    break

            del best_max_val, weight_t, x, weight, self.sampled_inputs[
                name], w_all, best_max_val_all
            paddle.device.cuda.empty_cache()
