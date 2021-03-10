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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def gan_loss(gan_mode, prediction, target_is_real, for_discriminator=True):
    if target_is_real:
        label = paddle.ones(shape=prediction.shape, dtype='float32')
    else:
        label = paddle.zeros(shape=prediction.shape, dtype='float32')
    if gan_mode == 'lsgan':
        loss = F.mse_loss(prediction, label)
    elif gan_mode == 'vanilla':
        loss = F.binary_cross_entropy_with_logits(prediction, label)
    elif gan_mode == 'wgangp':
        pass
    elif gan_mode == 'hinge':
        zero = paddle.zeros(shape=prediction.shape, dtype='float32')
        if for_discriminator:
            if target_is_real:
                minval = paddle.minimum(prediction - 1., zero)
                loss = -1. * paddle.mean(minval)
            else:
                minval = paddle.minimum(-1. * prediction - 1., zero)
                loss = -1. * paddle.mean(minval)
        else:
            assert target_is_real
            loss = -1. * paddle.mean(prediction)
    else:
        raise NotImplementedError('gan mode %s not implemented' % gan_mode)
    return loss


def recon_loss(mode, prediction, label):
    if mode == 'l1':
        loss = paddle.mean(paddle.abs(prediction - label))
    elif mode == 'l2':
        loss = F.mse_loss(prediction, label)
    elif mode == 'smooth_l1':
        loss = paddle.mean(F.smooth_l1_loss(prediction, label))
    elif mode == 'vgg':
        pass
    else:
        raise NotImplementedError('Unknown reconstruction loss type [%s]!' %
                                  mode)
    return loss
