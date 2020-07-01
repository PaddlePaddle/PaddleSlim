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

import paddle.fluid as fluid


def gan_loss(gan_mode, prediction, target_is_real, for_discriminator=True):
    if target_is_real:
        label = fluid.layers.fill_constant(
            shape=fluid.layers.shape(prediction), value=1.0, dtype='float32')
    else:
        label = fluid.layers.fill_constant(
            shape=fluid.layers.shape(prediction), value=0.0, dtype='float32')
    if gan_mode == 'lsgan':
        loss = fluid.layers.mse_loss(prediction, label)
    elif gan_mode == 'vanilla':
        loss = fluid.layers.sigmoid_cross_entropy_with_logits(prediction,
                                                              label)
    elif gan_mode == 'wgangp':
        pass
    elif gan_mode == 'hinge':
        zero = fluid.layers.fill_constant(
            shape=fluid.layers.shape(prediction), value=0.0, dtype='float32')
        if for_discriminator:
            if target_is_real:
                minval = fluid.layers.elementwise_min(prediction - 1., zero)
                loss = -1. * fluid.layers.reduce_mean(minval)
            else:
                minval = fluid.layers.elementwise_min(-1. * prediction - 1.,
                                                      zero)
                loss = -1. * fluid.layers.reduce_mean(minval)
        else:
            assert target_is_real
            loss = -1. * fluid.layers.reduce_mean(prediction)
    else:
        raise NotImplementedError('gan mode %s not implemented' % gan_mode)
    return loss


def recon_loss(mode, prediction, label):
    if mode == 'l1':
        loss = fluid.layers.reduce_mean(
            fluid.layers.elementwise_sub(
                prediction, label, act='abs'))
    elif mode == 'l2':
        loss = fluid.layers.mse_loss(prediction, label)
    elif mode == 'smooth_l1':
        loss = fluid.layers.reduce_mean(
            fluid.layers.smooth_l1(prediction, label))
    elif mode == 'vgg':
        pass
    else:
        raise NotImplementedError('Unknown reconstruction loss type [%s]!' %
                                  mode)
    return loss
