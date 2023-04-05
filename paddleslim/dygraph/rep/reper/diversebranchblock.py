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

# This code is referenced from: https://github.com/DingXiaoH/DiverseBranchBlock/blob/main/diversebranchblock.py

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr

from .base import BaseConv2DReper, ConvBNLayer

__all__ = ["DiverseBranchBlock"]


class IdentityBasedConv1x1(nn.Conv2D):
    def __init__(self,
                 channels,
                 groups=1,
                 weight_attr=ParamAttr(
                     initializer=nn.initializer.Constant(0.0))):
        super(IdentityBasedConv1x1, self).__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=False)

        assert channels % groups == 0
        input_dim = channels // groups
        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = paddle.to_tensor(id_value)
        self.groups = groups

    def forward(self, input):
        kernel = self.weight + self.id_tensor
        result = F.conv2d(
            input, kernel, None, stride=1, padding=0, groups=self.groups)
        return result

    def get_actual_kernel(self):
        return self.weight + self.id_tensor


class BNAndPadLayer(nn.Layer):
    def __init__(self, pad_pixels, num_features, eps=1e-5, momentum=0.1):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2D(num_features, momentum, eps)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            pad_values = self.bn.bias - self.bn._mean * self.bn.weight / paddle.sqrt(
                self.bn._variance + self.bn._epsilon)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.reshape((1, -1, 1, 1))
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def _mean(self):
        return self.bn._mean

    @property
    def _variance(self):
        return self.bn._variance

    @property
    def _epsilon(self):
        return self.bn._epsilon


class DiverseBranchBlock(BaseConv2DReper):
    """
    An instance of the DBB module, which replaces the conv-bn layer in the network.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 padding=None,
                 internal_channels_1x1_3x3=None):
        super(DiverseBranchBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding)

        # kxk branch
        self.dbb_origin = ConvBNLayer(
            in_channels, out_channels, kernel_size, stride, groups=groups)

        # 1x1-avg branch
        self.dbb_avg = nn.Sequential()
        if groups < out_channels:
            self.dbb_avg.add_sublayer('conv',
                                      nn.Conv2D(
                                          in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          groups=groups,
                                          bias_attr=False))
            self.dbb_avg.add_sublayer('bn',
                                      BNAndPadLayer(
                                          pad_pixels=self.padding,
                                          num_features=out_channels))
            self.dbb_avg.add_sublayer('avg',
                                      nn.AvgPool2D(
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=0))
        else:
            self.dbb_avg.add_sublayer('avg',
                                      nn.AvgPool2D(
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=self.padding))
        self.dbb_avg.add_sublayer('avgbn', nn.BatchNorm2D(out_channels))

        # 1x1 branch
        if groups < out_channels:
            self.dbb_1x1 = ConvBNLayer(
                in_channels, out_channels, 1, stride, groups=groups)

        # 1x1-kxk branch
        if internal_channels_1x1_3x3 is None:
            # For mobilenet, it is better to have 2X internal channels
            internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels
        self.dbb_1x1_kxk = nn.Sequential()
        if internal_channels_1x1_3x3 == in_channels:
            self.dbb_1x1_kxk.add_sublayer('idconv1',
                                          IdentityBasedConv1x1(
                                              channels=in_channels,
                                              groups=groups))
        else:
            self.dbb_1x1_kxk.add_sublayer(
                'conv1',
                nn.Conv2D(
                    in_channels=in_channels,
                    out_channels=internal_channels_1x1_3x3,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=groups,
                    bias_attr=False))
        self.dbb_1x1_kxk.add_sublayer(
            'bn1',
            BNAndPadLayer(
                pad_pixels=self.padding,
                num_features=internal_channels_1x1_3x3))
        self.dbb_1x1_kxk.add_sublayer('conv2',
                                      nn.Conv2D(
                                          in_channels=internal_channels_1x1_3x3,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=0,
                                          groups=groups,
                                          bias_attr=False))
        self.dbb_1x1_kxk.add_sublayer('bn2', nn.BatchNorm2D(out_channels))

    def _fuse_bn(self, kernel, bn):
        running_mean = bn._mean
        running_var = bn._variance
        gamma = bn.weight
        beta = bn.bias
        eps = bn._epsilon

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))

        return kernel * t, beta - running_mean * gamma / std

    def _fuse_1x1_kxk(self, k1, b1, k2, b2, groups):
        if groups == 1:
            k = F.conv2d(k2, k1.transpose((1, 0, 2, 3)))
            b_hat = (k2 * b1.reshape((1, -1, 1, 1))).sum((1, 2, 3))
        else:
            k_slices = []
            b_slices = []
            k1_T = k1.transpose((1, 0, 2, 3))
            k1_group_width = k1.shape[0] // groups
            k2_group_width = k2.shape[0] // groups
            for g in range(groups):
                k1_T_slice = k1_T[:, g * k1_group_width:(
                    g + 1) * k1_group_width, :, :]
                k2_slice = k2[g * k2_group_width:(g + 1
                                                  ) * k2_group_width, :, :, :]
                k_slices.append(F.conv2d(k2_slice, k1_T_slice))
                b_slices.append(
                    (k2_slice *
                     b1[g * k1_group_width:(g + 1) * k1_group_width].reshape(
                         (1, -1, 1, 1))).sum((1, 2, 3)))
            k = paddle.concat(k_slices)
            b_hat = paddle.concat(b_slices)
        return k, b_hat + b2

    def _fuse_avg(self, channels, kernel_size, groups):
        input_dim = channels // groups
        k = paddle.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels),
          np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size**2
        return k

    # This has not been tested with non-square kernels (kernel.size(2) != kernel.size(3)) nor even-size kernels
    def _fuse_multiscale(self, kernel, target_kernel_size):
        H_pixels_to_pad = (target_kernel_size - kernel.shape[2]) // 2
        W_pixels_to_pad = (target_kernel_size - kernel.shape[3]) // 2
        return F.pad(kernel, [
            H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad
        ])

    def _get_equivalent_kernel_bias(self):
        k_origin, b_origin = self._fuse_bn(self.dbb_origin.conv.weight,
                                           self.dbb_origin.bn)

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = self._fuse_bn(self.dbb_1x1.conv.weight,
                                         self.dbb_1x1.bn)
            k_1x1 = self._fuse_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = self._fuse_bn(
            k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = self._fuse_bn(
            self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = self._fuse_1x1_kxk(
            k_1x1_kxk_first,
            b_1x1_kxk_first,
            k_1x1_kxk_second,
            b_1x1_kxk_second,
            groups=self.groups)

        k_avg = self._fuse_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = self._fuse_bn(
            k_avg, self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = self._fuse_bn(
                self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = self._fuse_1x1_kxk(
                k_1x1_avg_first,
                b_1x1_avg_first,
                k_1x1_avg_second,
                b_1x1_avg_second,
                groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return sum([k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged]), sum(
            [b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged])

    def convert_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self._get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias_attr=True)
        self.dbb_reparam.weight.set_value(kernel)
        self.dbb_reparam.bias.set_value(bias)
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):

        if hasattr(self, 'dbb_reparam'):
            return self.dbb_reparam(inputs)

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return out
