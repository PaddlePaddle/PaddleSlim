# coding: utf8
# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
import numpy as np
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import MSRA
import contextlib

name_scope = ""

__all__ = ['FastSCNN']


@contextlib.contextmanager
def scope(name):
    global name_scope
    bk = name_scope
    name_scope = name_scope + name + '_'
    yield
    name_scope = bk


def conv_bn_layer(input,
                  filter_size,
                  num_filters,
                  stride,
                  padding,
                  channels=None,
                  num_groups=1,
                  if_act=True,
                  name=None,
                  use_cudnn=True):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        groups=num_groups,
        act=None,
        use_cudnn=use_cudnn,
        param_attr=ParamAttr(name=name + '_weights'),
        bias_attr=False)
    bn_name = name + '_bn'
    bn = fluid.layers.batch_norm(
        input=conv,
        param_attr=ParamAttr(name=bn_name + "_scale"),
        bias_attr=ParamAttr(name=bn_name + "_offset"),
        moving_mean_name=bn_name + '_mean',
        moving_variance_name=bn_name + '_variance')
    if if_act:
        return fluid.layers.relu6(bn)
    else:
        return bn


def separate_conv(input, channel, stride, filter, dilation=1, act=None):
    param_attr = fluid.ParamAttr(
        name=name_scope + 'weights',
        regularizer=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0),
        initializer=fluid.initializer.TruncatedNormal(
            loc=0.0, scale=0.33))
    with scope('depthwise'):
        input = conv(
            input,
            input.shape[1],
            filter,
            stride,
            groups=input.shape[1],
            padding=(filter // 2) * dilation,
            dilation=dilation,
            use_cudnn=False,
            param_attr=param_attr)
        input = bn(input)
        if act: input = act(input)

    param_attr = fluid.ParamAttr(
        name=name_scope + 'weights',
        regularizer=None,
        initializer=fluid.initializer.TruncatedNormal(
            loc=0.0, scale=0.06))
    with scope('pointwise'):
        input = conv(
            input, channel, 1, 1, groups=1, padding=0, param_attr=param_attr)
        input = bn(input)
        if act: input = act(input)
    return input


def avg_pool(input, kernel, stride, padding=0):
    data = fluid.layers.pool2d(
        input,
        pool_size=kernel,
        pool_type='avg',
        pool_stride=stride,
        pool_padding=padding)
    return data


def bn_relu(data):
    return fluid.layers.relu(bn(data))


def relu(data):
    return fluid.layers.relu(data)


def conv(*args, **kargs):
    kargs['param_attr'] = name_scope + 'weights'
    if 'bias_attr' in kargs and kargs['bias_attr']:
        kargs['bias_attr'] = fluid.ParamAttr(
            name=name_scope + 'biases',
            regularizer=None,
            initializer=fluid.initializer.ConstantInitializer(value=0.0))
    else:
        kargs['bias_attr'] = False
    return fluid.layers.conv2d(*args, **kargs)


def avg_pool(input, kernel, stride, padding=0):
    data = fluid.layers.pool2d(
        input,
        pool_size=kernel,
        pool_type='avg',
        pool_stride=stride,
        pool_padding=padding)
    return data


def bn(*args, **kargs):
    bn_regularizer = fluid.regularizer.L2DecayRegularizer(
        regularization_coeff=0.0)
    with scope('BatchNorm'):
        return fluid.layers.batch_norm(
            *args,
            param_attr=fluid.ParamAttr(
                name=name_scope + 'gamma', regularizer=bn_regularizer),
            bias_attr=fluid.ParamAttr(
                name=name_scope + 'beta', regularizer=bn_regularizer),
            moving_mean_name=name_scope + 'moving_mean',
            moving_variance_name=name_scope + 'moving_variance',
            **kargs)


def learning_to_downsample(x,
                           dw_channels1=32,
                           dw_channels2=48,
                           out_channels=64):
    x = relu(bn(conv(x, dw_channels1, 3, 2)))
    with scope('dsconv1'):
        x = separate_conv(
            x, dw_channels2, stride=2, filter=3, act=fluid.layers.relu)
    with scope('dsconv2'):
        x = separate_conv(
            x, out_channels, stride=2, filter=3, act=fluid.layers.relu)
    return x


def shortcut(input, data_residual):
    return fluid.layers.elementwise_add(input, data_residual)


def dropout2d(input, prob, is_train=False):
    if not is_train:
        return input
    channels = input.shape[1]
    keep_prob = 1.0 - prob
    random_tensor = keep_prob + fluid.layers.uniform_random_batch_size_like(
        input, [-1, channels, 1, 1], min=0., max=1.)
    binary_tensor = fluid.layers.floor(random_tensor)
    output = input / keep_prob * binary_tensor
    return output


def inverted_residual_unit(input,
                           num_in_filter,
                           num_filters,
                           ifshortcut,
                           stride,
                           filter_size,
                           padding,
                           expansion_factor,
                           name=None):
    num_expfilter = int(round(num_in_filter * expansion_factor))

    channel_expand = conv_bn_layer(
        input=input,
        num_filters=num_expfilter,
        filter_size=1,
        stride=1,
        padding=0,
        num_groups=1,
        if_act=True,
        name=name + '_expand')

    bottleneck_conv = conv_bn_layer(
        input=channel_expand,
        num_filters=num_expfilter,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        num_groups=num_expfilter,
        if_act=True,
        name=name + '_dwise',
        use_cudnn=False)

    depthwise_output = bottleneck_conv

    linear_out = conv_bn_layer(
        input=bottleneck_conv,
        num_filters=num_filters,
        filter_size=1,
        stride=1,
        padding=0,
        num_groups=1,
        if_act=False,
        name=name + '_linear')

    if ifshortcut:
        out = shortcut(input=input, data_residual=linear_out)
        return out, depthwise_output
    else:
        return linear_out, depthwise_output


def inverted_blocks(input, in_c, t, c, n, s, name=None):
    first_block, depthwise_output = inverted_residual_unit(
        input=input,
        num_in_filter=in_c,
        num_filters=c,
        ifshortcut=False,
        stride=s,
        filter_size=3,
        padding=1,
        expansion_factor=t,
        name=name + '_1')

    last_residual_block = first_block
    last_c = c

    for i in range(1, n):
        last_residual_block, depthwise_output = inverted_residual_unit(
            input=last_residual_block,
            num_in_filter=last_c,
            num_filters=c,
            ifshortcut=True,
            stride=1,
            filter_size=3,
            padding=1,
            expansion_factor=t,
            name=name + '_' + str(i + 1))
    return last_residual_block, depthwise_output


def psp_module(input, out_features):
    # Pyramid Scene Parsing 金字塔池化模块
    # 输入：backbone输出的特征
    # 输出：对输入进行不同尺度pooling, 卷积操作后插值回原始尺寸，并concat
    #       最后进行一个卷积及BN操作

    cat_layers = []
    sizes = (1, 2, 3, 6)
    h, w = input.shape[2:]
    for size in sizes:
        psp_name = "psp" + str(size)
        with scope(psp_name):
            pool = fluid.layers.adaptive_pool2d(
                input,
                pool_size=[size, size],
                pool_type='avg',
                name=psp_name + '_adapool')
            data = conv(
                pool,
                out_features,
                filter_size=1,
                bias_attr=False,
                name=psp_name + '_conv')
            data_bn = bn(data, act='relu')
            interp = fluid.layers.resize_bilinear(
                data_bn,
                out_shape=input.shape[2:],
                name=psp_name + '_interp',
                align_mode=0)
        cat_layers.append(interp)
    cat_layers = [input] + cat_layers
    out = fluid.layers.concat(cat_layers, axis=1, name='psp_cat')
    return out


class FeatureFusionModule():
    """Feature fusion module
    """

    def __init__(self,
                 higher_in_channels,
                 lower_in_channels,
                 out_channels,
                 scale_factor=4):
        self.higher_in_channels = higher_in_channels
        self.lower_in_channels = lower_in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

    def net(self, higher_res_feature, lower_res_feature):
        lower_res_feature = fluid.layers.resize_bilinear(
            lower_res_feature, scale=4, align_mode=0)

        with scope('dwconv'):
            lower_res_feature = relu(
                bn(conv(lower_res_feature, self.out_channels, 1)))
        with scope('conv_lower_res'):
            lower_res_feature = bn(
                conv(
                    lower_res_feature, self.out_channels, 1, bias_attr=True))
        with scope('conv_higher_res'):
            higher_res_feature = bn(
                conv(
                    higher_res_feature, self.out_channels, 1, bias_attr=True))
        out = higher_res_feature + lower_res_feature

        return relu(out)


class GlobalFeatureExtractor():
    """Global feature extractor module"""

    def __init__(self,
                 in_channels=64,
                 block_channels=(64, 96, 128),
                 out_channels=128,
                 t=6,
                 num_blocks=(3, 3, 3)):
        self.in_channels = in_channels
        self.block_channels = block_channels
        self.out_channels = out_channels
        self.t = t
        self.num_blocks = num_blocks

    def net(self, x):
        x, _ = inverted_blocks(x, self.in_channels, self.t,
                               self.block_channels[0], self.num_blocks[0], 2,
                               'inverted_block_1')
        x, _ = inverted_blocks(x, self.block_channels[0], self.t,
                               self.block_channels[1], self.num_blocks[1], 2,
                               'inverted_block_2')
        x, _ = inverted_blocks(x, self.block_channels[1], self.t,
                               self.block_channels[2], self.num_blocks[2], 1,
                               'inverted_block_3')
        x = psp_module(x, self.block_channels[2] // 4)
        with scope('out'):
            x = relu(bn(conv(x, self.out_channels, 1)))
        return x


class Classifer():
    """Classifer"""

    def __init__(self, dw_channels, class_dim, stride=1):
        self.dw_channels = dw_channels
        self.class_dim = class_dim
        self.stride = stride

    def net(self, x):
        with scope('dsconv1'):
            x = separate_conv(
                x,
                self.dw_channels,
                stride=self.stride,
                filter=3,
                act=fluid.layers.relu)
        with scope('dsconv2'):
            x = separate_conv(
                x,
                self.dw_channels,
                stride=self.stride,
                filter=3,
                act=fluid.layers.relu)
        x = dropout2d(x, 0.1, is_train=True)
        x = conv(x, self.class_dim, 1, bias_attr=True)
        return x


def aux_layer(x, class_dim):
    x = relu(bn(conv(x, 32, 3, padding=1)))
    x = dropout2d(x, 0.1, is_train=True)
    with scope('logit'):
        x = conv(x, class_dim, 1, bias_attr=True)
    return x


class FastSCNN():
    def net(self, input, class_dim):
        size = input.shape[2:]
        classifier = Classifer(128, class_dim)
        global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128],
                                                          128, 6, [3, 3, 3])
        feature_fusion = FeatureFusionModule(64, 128, 128)
        with scope('learning_to_downsample'):
            higher_res_features = learning_to_downsample(input, 32, 48, 64)
        with scope('global_feature_extractor'):
            lower_res_feature = global_feature_extractor.net(
                higher_res_features)
        with scope('feature_fusion'):
            x = feature_fusion.net(higher_res_features, lower_res_feature)
        with scope('classifier'):
            logit = classifier.net(x)
            logit = fluid.layers.resize_bilinear(logit, size, align_mode=0)

        output = fluid.layers.fc(input=logit,
                                 size=class_dim,
                                 act='softmax',
                                 param_attr=ParamAttr(
                                     initializer=MSRA(), name="fc_weights"),
                                 bias_attr=ParamAttr(name="fc_offset"))
        return output
