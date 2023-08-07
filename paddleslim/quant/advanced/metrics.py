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


def mse_loss(y_pred, y_real, reduction='mean'):
    if y_pred.shape != y_real.shape:
        raise ValueError(
            f'Can not compute mse loss for tensors with different shape. '
            f'({y_pred.shape} and {y_real.shape})')

    mse = (y_pred - y_real)**2

    if reduction == 'mean':
        return paddle.mean(mse)
    elif reduction == 'sum':
        return paddle.sum(mse)
    elif reduction == 'none':
        return mse
    else:
        raise ValueError(f'Unsupported reduction method.')


def snr_loss(y_pred, y_real, reduction='mean'):
    if y_pred.shape != y_real.shape:
        raise ValueError(
            f'Can not compute snr loss for tensors with different shape. '
            f'({y_pred.shape} and {y_real.shape})')
    reduction = str(reduction).lower()

    y_pred = y_pred.flatten(start_axis=1)
    y_real = y_real.flatten(start_axis=1)

    noise_power = paddle.pow(y_pred - y_real, 2).sum(axis=-1)
    signal_power = paddle.pow(y_real, 2).sum(axis=-1)
    snr = (noise_power) / (signal_power + 1e-7)

    if reduction == 'mean':
        return paddle.mean(snr)
    elif reduction == 'sum':
        return paddle.sum(snr)
    elif reduction == 'none':
        return snr
    else:
        raise ValueError(f'Unsupported reduction method.')
