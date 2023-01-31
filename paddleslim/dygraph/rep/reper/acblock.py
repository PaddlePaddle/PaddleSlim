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

import numpy as np

import paddle
import paddle.nn as nn

from .base import BaseConv2DReper, ConvBNLayer

__all__ = ["ACBlock"]


class ACBlock(BaseConv2DReper):
    """
    An instance of the ACBlock module, which replaces the conv-bn layer in the network.
    Refer from Paper: https://arxiv.org/abs/1908.03930.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 padding=None):
        super(ACBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding)
        if self.padding - self.kernel_size // 2 >= 0:
            self.crop = 0
            # Compared to the KxK layer, the padding of the 1xK layer and Kx1 layer should be adjust to align the sliding windows (Fig 2 in the paper)
            hor_padding = [self.padding - self.kernel_size // 2, self.padding]
            ver_padding = [self.padding, self.padding - self.kernel_size // 2]
        else:
            # A negative "padding" (padding - kernel_size//2 < 0, which is not a common use case) is cropping.
            # Since nn.Conv2D does not support negative padding, we implement it manually
            self.crop = self.kernel_size // 2 - self.padding
            hor_padding = [0, self.padding]
            ver_padding = [self.padding, 0]

        # kxk square branch
        self.square_branch = ConvBNLayer(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            groups=self.groups,
            padding=self.padding)

        # kx1 vertical branch
        self.ver_branch = ConvBNLayer(
            self.in_channels,
            self.out_channels, (self.kernel_size, 1),
            self.stride,
            groups=self.groups,
            padding=ver_padding)

        # 1xk horizontal branch
        self.hor_branch = ConvBNLayer(
            self.in_channels,
            self.out_channels, (1, self.kernel_size),
            self.stride,
            groups=self.groups,
            padding=hor_padding)

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.shape[2]
        asym_w = asym_kernel.shape[3]
        square_h = square_kernel.shape[2]
        square_w = square_kernel.shape[3]
        square_kernel[:, :, square_h // 2 - asym_h // 2:square_h // 2 -
                      asym_h // 2 + asym_h, square_w // 2 - asym_w // 2:
                      square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

    def _fuse_bn(self, kernel, bn):
        running_mean = bn._mean
        running_var = bn._variance
        gamma = bn.weight
        beta = bn.bias
        eps = bn._epsilon

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))

        return kernel * t, beta - running_mean * gamma / std

    def _get_equivalent_kernel_bias(self):
        hor_k, hor_b = self._fuse_bn(self.hor_branch.conv.weight,
                                     self.hor_branch.bn)
        ver_k, ver_b = self._fuse_bn(self.ver_branch.conv.weight,
                                     self.ver_branch.bn)
        square_k, square_b = self._fuse_bn(self.square_branch.conv.weight,
                                           self.square_branch.bn)
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        return square_k, hor_b + ver_b + square_b

    def convert_to_deploy(self):
        if hasattr(self, 'fused_branch'):
            return
        kernel, bias = self._get_equivalent_kernel_bias()
        self.fused_branch = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias_attr=True)
        self.fused_branch.weight.set_value(kernel)
        self.fused_branch.bias.set_value(bias)
        self.__delattr__('ver_branch')
        self.__delattr__('hor_branch')
        self.__delattr__('square_branch')

    def forward(self, input):
        if hasattr(self, 'fused_branch'):
            return self.fused_branch(input)

        out = self.square_branch(input)
        if self.crop > 0:
            ver_input = input[:, :, :, self.crop:-self.crop]
            hor_input = input[:, :, self.crop:-self.crop, :]
        else:
            ver_input = input
            hor_input = input
        out += self.ver_branch(ver_input)
        out += self.hor_branch(hor_input)
        return out
