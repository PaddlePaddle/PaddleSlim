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

import paddle
from paddle.nn import Conv2D
from paddle.nn import BatchNorm
from paddle.nn.initializer import Constant, KaimingUniform


OPS = {
    'none':
    lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3':
    lambda C, stride, affine: paddle.nn.AvgPool2D(
        3,
        stride=stride,
        padding=1),
    'max_pool_3x3':
    lambda C, stride, affine: paddle.nn.MaxPool2D(
        3,
        stride=stride,
        padding=1),
    'skip_connect':
    lambda C, stride, affine: Identity()
    if stride == 1 else FactorizedReduce(C, C, affine),
    'sep_conv_3x3':
    lambda C, stride, affine: SepConv(C, C, 3, stride, 1,
                                                       affine),
    'sep_conv_5x5':
    lambda C, stride, affine: SepConv(C, C, 5, stride, 2,
                                                       affine),
    'sep_conv_7x7':
    lambda C, stride, affine: SepConv(C, C, 7, stride, 3,
                                                       affine),
    'dil_conv_3x3':
    lambda C, stride, affine: DilConv(C, C, 3, stride, 2,
                                                       2, affine),
    'dil_conv_5x5':
    lambda C, stride, affine: DilConv(C, C, 5, stride, 4,
                                                       2, affine),
    'conv_7x1_1x7':
    lambda C, stride, affine: Conv_7x1_1x7(
        C, C, stride, affine),
}


def bn_param_config(affine=False):
    gama = paddle.ParamAttr(initializer=Constant(value=1), trainable=affine)
    beta = paddle.ParamAttr(initializer=Constant(value=0), trainable=affine)
    return gama, beta


class Zero(paddle.nn.Layer):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride
        self.pool = paddle.nn.MaxPool2D(1, stride=2)

    def forward(self, x):
        pooled = self.pool(x)
        x = paddle.zeros_like(x) if self.stride == 1 else paddle.zeros_like(
            pooled)
        return x


class Identity(paddle.nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FactorizedReduce(paddle.nn.Layer):
    def __init__(self, c_in, c_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert c_out % 2 == 0
        self.conv1 = Conv2D(
            num_channels=c_in,
            num_filters=c_out // 2,
            filter_size=1,
            stride=2,
            padding=0,
            param_attr=paddle.ParamAttr(initializer=KaimingUniform()),
            bias_attr=False)
        self.conv2 = Conv2D(
            num_channels=c_in,
            num_filters=c_out // 2,
            filter_size=1,
            stride=2,
            padding=0,
            param_attr=paddle.ParamAttr(initializer=KaimingUniform()),
            bias_attr=False)
        gama, beta = bn_param_config(affine)
        self.bn = BatchNorm(num_channels=c_out, param_attr=gama, bias_attr=beta)

    def forward(self, x):
        x = paddle.nn.functional.relu(x)
        out = paddle.concat(
            input=[self.conv1(x), self.conv2(x[:, :, 1:, 1:])], axis=1)
        out = self.bn(out)
        return out


class SepConv(paddle.nn.Layer):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.conv1 = Conv2D(
            num_channels=c_in,
            num_filters=c_in,
            filter_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=c_in,
            use_cudnn=False,
            param_attr=paddle.ParamAttr(initializer=KaimingUniform()),
            bias_attr=False)
        self.conv2 = Conv2D(
            num_channels=c_in,
            num_filters=c_in,
            filter_size=1,
            stride=1,
            padding=0,
            param_attr=paddle.ParamAttr(initializer=KaimingUniform()),
            bias_attr=False)
        gama, beta = bn_param_config(affine)
        self.bn1 = BatchNorm(num_channels=c_in, param_attr=gama, bias_attr=beta)
        self.conv3 = Conv2D(
            num_channels=c_in,
            num_filters=c_in,
            filter_size=kernel_size,
            stride=1,
            padding=padding,
            groups=c_in,
            use_cudnn=False,
            param_attr=paddle.ParamAttr(initializer=KaimingUniform()),
            bias_attr=False)
        self.conv4 = Conv2D(
            num_channels=c_in,
            num_filters=c_out,
            filter_size=1,
            stride=1,
            padding=0,
            param_attr=paddle.ParamAttr(initializer=KaimingUniform()),
            bias_attr=False)
        gama, beta = bn_param_config(affine)
        self.bn2 = BatchNorm(
            num_channels=c_out, param_attr=gama, bias_attr=beta)

    def forward(self, x):
        x = paddle.nn.functional.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        bn1 = self.bn1(x)
        x = paddle.nn.functional.relu(bn1)
        x = self.conv3(x)
        x = self.conv4(x)
        bn2 = self.bn2(x)
        return bn2


class DilConv(paddle.nn.Layer):
    def __init__(self,
                 c_in,
                 c_out,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 affine=True):
        super(DilConv, self).__init__()
        self.conv1 = Conv2D(
            num_channels=c_in,
            num_filters=c_in,
            filter_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=c_in,
            use_cudnn=False,
            param_attr=paddle.ParamAttr(initializer=KaimingUniform()),
            bias_attr=False)
        self.conv2 = Conv2D(
            num_channels=c_in,
            num_filters=c_out,
            filter_size=1,
            padding=0,
            param_attr=paddle.ParamAttr(initializer=KaimingUniform()),
            bias_attr=False)
        gama, beta = bn_param_config(affine)
        self.bn1 = BatchNorm(
            num_channels=c_out, param_attr=gama, bias_attr=beta)

    def forward(self, x):
        x = paddle.nn.functional.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.bn1(x)
        return out


class Conv_7x1_1x7(paddle.nn.Layer):
    def __init__(self, c_in, c_out, stride, affine=True):
        super(Conv_7x1_1x7, self).__init__()
        self.conv1 = Conv2D(
            num_channels=c_in,
            num_filters=c_out,
            filter_size=(1, 7),
            padding=(0, 3),
            param_attr=paddle.ParamAttr(initializer=KaimingUniform()),
            bias_attr=False)
        self.conv2 = Conv2D(
            num_channels=c_in,
            num_filters=c_out,
            filter_size=(7, 1),
            padding=(3, 0),
            param_attr=paddle.ParamAttr(initializer=KaimingUniform()),
            bias_attr=False)
        gama, beta = bn_param_config(affine)
        self.bn1 = BatchNorm(
            num_channels=c_out, param_attr=gama, bias_attr=beta)

    def forward(self, x):
        x = paddle.nn.functional.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.bn1(x)
        return out


class ReLUConvBN(paddle.nn.Layer):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.conv = Conv2D(
            num_channels=c_in,
            num_filters=c_out,
            filter_size=kernel_size,
            stride=stride,
            padding=padding,
            param_attr=paddle.ParamAttr(initializer=KaimingUniform()),
            bias_attr=False)
        gama, beta = bn_param_config(affine)
        self.bn = BatchNorm(num_channels=c_out, param_attr=gama, bias_attr=beta)

    def forward(self, x):
        x = paddle.nn.functional.relu(x)
        x = self.conv(x)
        out = self.bn(x)
        return out
