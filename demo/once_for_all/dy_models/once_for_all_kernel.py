# coding:utf-8
#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
import numpy as np

from dy_models.func_conv import conv2d
#from func_conv import conv2d
import pdb


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class OFA_kernel(fluid.dygraph.Layer):
    def __init__(self,
                 scale=1.0,
                 model_name='large',
                 token=[],
                 class_dim=1000,
                 ofa_mode=None,
                 trainable_besides_trans=False):
        super(OFA_kernel, self).__init__()
        # Parsing token
        assert len(token) >= 45
        self.kernel_size_lis = tuple(token[:20])
        self.exp_lis = tuple(token[20:40])
        self.depth_lis = tuple(token[40:45])

        # Hyperparameter
        self.scale = scale
        self.inplanes = 16
        self.class_dim = class_dim
        self.ofa_mode = ofa_mode
        self.trainable_besides_trans = trainable_besides_trans
        if model_name == "large":
            # The search space is the last five digits
            self.cfg_channel = (16, 24, 40, 80, 112, 160)
            self.cfg_stride = (1, 2, 2, 2, 1, 2)
            self.cfg_se = (False, False, True, False, True, True)
            self.cfg_act = ('relu', 'relu', 'relu', 'hard_swish', 'hard_swish',
                            'hard_swish')
            self.cls_ch_squeeze = 960
            self.cls_ch_expand = 1280
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        # conv1
        self.conv1 = DynamicConvBnLayer(
            in_c=3,
            out_c=make_divisible(self.inplanes * self.scale),
            filter_size=3,
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act='hard_swish',
            ofa_mode=None,
            trainable_besides_trans=self.trainable_besides_trans,
            name='conv1')
        self.inplanes = make_divisible(self.inplanes * self.scale)

        # conv2
        num_mid_filter = make_divisible(self.inplanes * self.scale)
        _num_out_filter = self.cfg_channel[0]
        num_out_filter = make_divisible(self.scale * _num_out_filter)
        self.conv2 = ResidualUnit(
            in_c=self.inplanes,
            mid_c=num_out_filter,
            out_c=num_out_filter,
            filter_size=3,
            stride=self.cfg_stride[0],
            use_se=self.cfg_se[0],
            act=self.cfg_act[0],
            ofa_mode=None,
            trainable_besides_trans=self.trainable_besides_trans,
            name='conv2')
        self.inplanes = make_divisible(self.cfg_channel[0] * self.scale)

        # conv_blocks
        i = 3
        self.conv_blocks = []
        for depth_id in range(len(self.depth_lis)):
            for repeat_time in range(self.depth_lis[depth_id]):
                num_mid_filter = make_divisible(
                    self.scale * _num_out_filter *
                    self.exp_lis[depth_id * 4 + repeat_time])
                _num_out_filter = self.cfg_channel[depth_id + 1]
                num_out_filter = make_divisible(self.scale * _num_out_filter)
                stride = self.cfg_stride[depth_id +
                                         1] if repeat_time == 0 else 1
                self.conv_blocks.append(
                    ResidualUnit(
                        in_c=self.inplanes,
                        mid_c=num_mid_filter,
                        out_c=num_out_filter,
                        filter_size=self.kernel_size_lis[depth_id * 4 +
                                                         repeat_time],
                        stride=stride,
                        use_se=self.cfg_se[depth_id + 1],
                        act=self.cfg_act[depth_id + 1],
                        ofa_mode=self.ofa_mode,
                        trainable_besides_trans=self.trainable_besides_trans,
                        name='conv' + str(i)))
                self.add_sublayer(
                    sublayer=self.conv_blocks[-1], name='conv' + str(i))
                self.inplanes = make_divisible(self.scale *
                                               self.cfg_channel[depth_id + 1])
                i += 1

        # last_second_conv
        self.last_second_conv = DynamicConvBnLayer(
            in_c=self.inplanes,
            out_c=make_divisible(self.cls_ch_squeeze * self.scale),
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            act='hard_swish',
            ofa_mode=None,
            trainable_besides_trans=self.trainable_besides_trans,
            name='last_second_conv')

        # global_avg_pool
        self.global_avg_pool = fluid.dygraph.Pool2D(
            pool_type='avg', global_pooling=True, use_cudnn=False)

        # last_conv
        self.last_conv = fluid.dygraph.Conv2D(
            num_channels=make_divisible(self.cls_ch_squeeze * self.scale),
            num_filters=self.cls_ch_expand,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            param_attr=ParamAttr(
                name='last_conv_weights',
                trainable=self.trainable_besides_trans),
            bias_attr=ParamAttr(
                name='last_conv_bias', trainable=self.trainable_besides_trans))

        # fc
        self.fc = fluid.dygraph.Linear(
            input_dim=self.cls_ch_expand,
            output_dim=self.class_dim,
            param_attr=ParamAttr(
                name='fc_weights', trainable=self.trainable_besides_trans),
            bias_attr=ParamAttr(
                name='fc_bias', trainable=self.trainable_besides_trans))

    def forward(self,
                inputs,
                label=None,
                dy_token=[],
                dy_trainable=[],
                dropout_prob=0.2):
        #pdb.set_trace()
        x = self.conv1(inputs)
        x = self.conv2(x)
        if self.ofa_mode == 'kernel':
            assert len(dy_token) >= sum(self.depth_lis)
            assert len(dy_trainable) >= sum(self.depth_lis)
            count = 0
            i = -1
            for depth_id in range(len(self.depth_lis)):
                for repeat_time in range(self.depth_lis[depth_id]):
                    i += 1
                    if self.kernel_size_lis[depth_id * 4 +
                                            repeat_time] != dy_token[i]:
                        count += 1
            if count != 1:
                # print('WARNING: According to the original paper, only one filter should be transformed in each iteration, but the current iteration transforms {} filter(s).'.format(count))
                pass
        else:
            dy_token = [None for i in range(sum(self.depth_lis))]
            dy_trainable = [True for i in range(sum(self.depth_lis))]
        for i in range(len(self.conv_blocks)):
            x = self.conv_blocks[i](x, dy_token[i], dy_trainable[i])
        #pdb.set_trace()
        x = self.last_second_conv(x)
        x = self.global_avg_pool(x)
        x = self.last_conv(x)
        x = fluid.layers.hard_swish(x)
        x = fluid.layers.dropout(x=x, dropout_prob=dropout_prob)
        x = fluid.layers.squeeze(x, axes=[])
        x = self.fc(x)

        if label:
            acc1 = fluid.layers.accuracy(input=x, label=label)
            acc5 = fluid.layers.accuracy(input=x, label=label, k=5)
            return x, acc1, acc5
        return x


class DynamicConvBnLayer(fluid.dygraph.Layer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1,
                 if_act=True,
                 act=None,
                 use_cudnn=True,
                 ofa_mode=None,
                 trainable_besides_trans=True,
                 name=''):
        super(DynamicConvBnLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.filter_size = filter_size
        self.stride = stride
        self.num_groups = num_groups
        self.ofa_mode = ofa_mode
        self.trainable_besides_trans = trainable_besides_trans
        self.use_cudnn = use_cudnn
        self.name = name

        self.conv = fluid.dygraph.Conv2D(
            num_channels=in_c,
            num_filters=out_c,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            param_attr=ParamAttr(
                trainable=self.trainable_besides_trans, name=name + "_weights"),
            bias_attr=False,
            use_cudnn=use_cudnn,
            act=None)
        self.bn = fluid.dygraph.BatchNorm(
            num_channels=out_c,
            act=None,
            param_attr=ParamAttr(
                name=name + "_bn" + "_scale",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0),
                trainable=self.trainable_besides_trans,
                learning_rate=1.0 if self.trainable_besides_trans else 0.0),
            bias_attr=ParamAttr(
                name=name + "_bn" + "_offset",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0),
                trainable=self.trainable_besides_trans,
                learning_rate=1.0 if self.trainable_besides_trans else 0.0),
            moving_mean_name=name + "_bn" + '_mean',
            moving_variance_name=name + "_bn" + '_variance',
            use_global_stats=True)

        if not self.ofa_mode:
            pass
        elif self.ofa_mode == 'kernel':
            if num_groups == 1:
                raise RuntimeError(
                    'OFA only supports depthwise convolution for kernel transformation operations.'
                )
            self.trans_block = []
            if filter_size >= 5:
                _init_np_array = np.eye(9)
                self.trans_block.append(
                    fluid.dygraph.Linear(
                        input_dim=9,
                        output_dim=9,
                        param_attr=ParamAttr(
                            initializer=fluid.initializer.NumpyArrayInitializer(
                                _init_np_array),
                            regularizer=fluid.regularizer.L2Decay(0.0),
                            name=name + "_transLinear_5to3"),
                        bias_attr=False))
                self.add_sublayer(
                    sublayer=self.trans_block[-1],
                    name=name + "_transLinear_5to3")
                self.bn_3x3 = fluid.dygraph.BatchNorm(
                    num_channels=out_c,
                    act=None,
                    param_attr=ParamAttr(
                        name=name + "_bn_3x3" + "_scale",
                        regularizer=fluid.regularizer.L2DecayRegularizer(
                            regularization_coeff=0.0)),
                    bias_attr=ParamAttr(
                        name=name + "_bn_3x3" + "_offset",
                        regularizer=fluid.regularizer.L2DecayRegularizer(
                            regularization_coeff=0.0)),
                    moving_mean_name=name + "_bn_3x3" + '_mean',
                    moving_variance_name=name + "_bn_3x3" + '_variance')
            if filter_size >= 7:
                _init_np_array = np.eye(25)
                self.trans_block.insert(
                    0,
                    fluid.dygraph.Linear(
                        input_dim=25,
                        output_dim=25,
                        param_attr=ParamAttr(
                            initializer=fluid.initializer.NumpyArrayInitializer(
                                _init_np_array),
                            regularizer=fluid.regularizer.L2Decay(0.0),
                            name=name + "_transLinear_7to5"),
                        bias_attr=False))
                self.add_sublayer(
                    sublayer=self.trans_block[0],
                    name=name + "_transLinear_7to5")
                self.bn_5x5 = fluid.dygraph.BatchNorm(
                    num_channels=out_c,
                    act=None,
                    param_attr=ParamAttr(
                        name=name + "_bn_5x5" + "_scale",
                        regularizer=fluid.regularizer.L2DecayRegularizer(
                            regularization_coeff=0.0)),
                    bias_attr=ParamAttr(
                        name=name + "_bn_5x5" + "_offset",
                        regularizer=fluid.regularizer.L2DecayRegularizer(
                            regularization_coeff=0.0)),
                    moving_mean_name=name + "_bn_5x5" + '_mean',
                    moving_variance_name=name + "_bn_5x5" + '_variance')
        else:
            raise NotImplementedError("OFA_mode [" + ofa_mode +
                                      "] is not implemented!")

    def forward(self, x, dy_filter_size=None, dy_trans_trainable=True):
        if self.ofa_mode == 'kernel':
            for i in range(len(self.trans_block)):
                self.trans_block[i].weight.trainable = dy_trans_trainable
            # print(self.trans_block[0].weight.trainable)
            assert dy_filter_size in [3, 5, 7, None]
            if dy_filter_size is None or dy_filter_size == self.filter_size:
                x = self.conv(x)
            elif dy_filter_size > self.filter_size:
                raise RuntimeError(
                    'The new filter size should be less than or equal to the size of the original model.'
                )
            else:
                kernel_weight_np = self.conv.weight.numpy().copy()
                kernel_weight = fluid.dygraph.to_variable(
                    kernel_weight_np, zero_copy=False)
                kernel_weight.stop_gradient = False
                _batch_size, _channel = kernel_weight.shape[
                    0], kernel_weight.shape[1]
                i = 0
                while dy_filter_size < kernel_weight.shape[-1]:
                    kernel_weight = kernel_weight[:, :, 1:-1, 1:-1]
                    kernel_weight = fluid.layers.reshape(
                        kernel_weight, [_batch_size, _channel, -1])
                    kernel_weight = fluid.layers.reshape(
                        kernel_weight, [-1, kernel_weight.shape[-1]])
                    kernel_weight = self.trans_block[i](kernel_weight)
                    kernel_weight = fluid.layers.reshape(
                        kernel_weight, [_batch_size, _channel, -1])
                    _new_filter_size = int(kernel_weight.shape[-1]**0.5)
                    kernel_weight = fluid.layers.reshape(kernel_weight, [
                        _batch_size, _channel, _new_filter_size,
                        _new_filter_size
                    ])
                    i += 1

                x = conv2d(
                    x,
                    weight=kernel_weight,
                    bias=None,
                    padding=int((dy_filter_size - 1) // 2),
                    stride=self.stride,
                    dilation=1,
                    groups=self.num_groups,
                    use_cudnn=self.use_cudnn,
                    act=None,
                    data_format='NCHW',
                    name=self.name + '_transConv')

                if self.filter_size == dy_filter_size:
                    x = self.bn(x)
                elif dy_filter_size == 5:
                    x = self.bn_5x5(x)
                elif dy_filter_size == 3:
                    x = self.bn_3x3(x)
                else:
                    print('ERROR')
                    exit()
        else:
            x = self.conv(x)
            x = self.bn(x)
        if self.if_act:
            if self.act == 'relu':
                x = fluid.layers.relu(x)
            elif self.act == 'hard_swish':
                x = fluid.layers.hard_swish(x)
            elif self.act == 'mish':
                x = x * fluid.layers.tanh(fluid.layers.softplus(x))
            else:
                print('The activation function is selected incorrectly.')
                exit()
        return x


class ResidualUnit(fluid.dygraph.Layer):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 use_se,
                 act=None,
                 ofa_mode=None,
                 trainable_besides_trans=True,
                 name=''):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.if_se = use_se

        self.expand_conv = DynamicConvBnLayer(
            in_c=in_c,
            out_c=mid_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
            ofa_mode=None,
            trainable_besides_trans=trainable_besides_trans,
            name=name + '_expand')
        self.bottleneck_conv = DynamicConvBnLayer(
            in_c=mid_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            num_groups=mid_c,
            if_act=True,
            act=act,
            use_cudnn=False,
            ofa_mode=ofa_mode,
            trainable_besides_trans=trainable_besides_trans,
            name=name + '_depthwise')
        if self.if_se:
            self.mid_se = SEModule(
                mid_c,
                trainable_besides_trans=trainable_besides_trans,
                name=name + '_SE')
        self.linear_conv = DynamicConvBnLayer(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None,
            ofa_mode=None,
            trainable_besides_trans=trainable_besides_trans,
            name=name + '_linear')

    def forward(self, inputs, dy_filter_size=None, dy_trans_trainable=True):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x, dy_filter_size, dy_trans_trainable)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = fluid.layers.elementwise_add(inputs, x)
        return x


class SEModule(fluid.dygraph.Layer):
    def __init__(self,
                 channel,
                 reduction=4,
                 trainable_besides_trans=True,
                 name=''):
        super(SEModule, self).__init__()
        self.avg_pool = fluid.dygraph.Pool2D(
            pool_type='avg', global_pooling=True, use_cudnn=False)
        self.conv1 = fluid.dygraph.Conv2D(
            num_channels=channel,
            num_filters=channel // reduction,
            filter_size=1,
            stride=1,
            padding=0,
            act='relu',
            param_attr=ParamAttr(
                name=name + "_weights1", trainable=trainable_besides_trans),
            bias_attr=ParamAttr(
                name=name + "_bias1", trainable=trainable_besides_trans))
        self.conv2 = fluid.dygraph.Conv2D(
            num_channels=channel // reduction,
            num_filters=channel,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(
                name=name + "_weights2", trainable=trainable_besides_trans),
            bias_attr=ParamAttr(
                name=name + "_bias2", trainable=trainable_besides_trans))

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = fluid.layers.hard_sigmoid(outputs, slope=0.2, offset=0.5)
        return fluid.layers.elementwise_mul(x=inputs, y=outputs, axis=0)


if __name__ == "__main__":
    # from calc_flops import summary
    import numpy as np
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = OFA_kernel(
            scale=1.0,
            model_name='large',
            token=[
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 3,
                3, 3, 3, 3, 4, 3, 4, 3, 6, 4, 4, 4, 4, 6, 3, 6, 4, 6, 3, 2, 3,
                3, 3, 4
            ],
            class_dim=1000,
            ofa_mode='kernel',
            trainable_besides_trans=False)

        img = np.random.uniform(0, 255, [8, 3, 224, 224]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        token = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 7, 5, 7]
        trainable = [
            False, False, False, False, False, False, False, False, False,
            False, False, True, False, False, False
        ]
        res = model(img, dy_token=token, dy_trainable=trainable)
        print(res.shape)
