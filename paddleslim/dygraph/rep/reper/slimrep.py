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
from paddle import ParamAttr
from paddle.regularizer import L2Decay
import paddle.nn as nn

from .base import BaseConv2DReper, ConvBNLayer

__all__ = ["SlimRepBlock"]


class SlimRepBlock(BaseConv2DReper):
    """
    An instance of the SlimRepBlock module, which replaces the conv-bn layer in the network.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 padding=None):
        super(SlimRepBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding)
        self.num_conv_branches = 1
        if not self.padding:
            self.padding = self.kernel_size // 2
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

        # Re-parameterizable skip connection
        self.rbr_skip = nn.BatchNorm2D(
            num_features=in_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0))
        ) if in_channels == out_channels and self.stride == 1 else None

        # Re-parameterizable conv branches
        self.rbr_conv = nn.LayerList()
        for _ in range(self.num_conv_branches):
            for kernel_size in range(self.kernel_size, 0, -2):
                self.rbr_conv.append(
                    ConvBNLayer(
                        self.in_channels,
                        self.out_channels,
                        kernel_size,
                        stride=self.stride,
                        groups=self.groups))

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

    def forward(self, x):
        # Inference mode forward pass.
        if hasattr(self, "reparam_conv"):
            return self.reparam_conv(x)

        # Multi-branched train-time forward pass.
        out = 0
        for rbr_conv in self.rbr_conv:
            out += rbr_conv(x)
        # Skip branch output
        if self.rbr_skip is not None:
            out += self.rbr_skip(x)

        if self.crop > 0:
            ver_input = x[:, :, :, self.crop:-self.crop]
            hor_input = x[:, :, self.crop:-self.crop, :]
        else:
            ver_input = x
            hor_input = x
        out += self.ver_branch(ver_input)
        out += self.hor_branch(hor_input)

        return out

    def convert_to_deploy(self):
        """
        Re-parameterize multi-branched architecture used at training
        time to obtain a plain CNN-like structure for inference.
        """
        if hasattr(self, 'reparam_conv'):
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(self.kernel_size - 1) // 2,
            groups=self.groups)
        self.reparam_conv.weight.set_value(kernel)
        self.reparam_conv.bias.set_value(bias)

        # Delete un-used branches
        self.__delattr__('rbr_conv')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')
        self.__delattr__('ver_branch')
        self.__delattr__('hor_branch')

    def _get_kernel_bias(self):
        """ 
        Method to obtain re-parameterized kernel and bias.
        """

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            _kernel = self._pad_tensor(_kernel, to_size=self.kernel_size)
            kernel_conv += _kernel
            bias_conv += _bias

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        kernel_final = kernel_conv + kernel_identity
        bias_final = bias_conv + bias_identity

        # get kx1 1xk branch
        hor_k, hor_b = self._fuse_bn_tensor(self.hor_branch)
        ver_k, ver_b = self._fuse_bn_tensor(self.ver_branch)
        self._add_to_square_kernel(kernel_final, hor_k)
        self._add_to_square_kernel(kernel_final, ver_k)

        bias_final += hor_b + ver_b

        return kernel_final, bias_final

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.shape[2]
        asym_w = asym_kernel.shape[3]
        square_h = square_kernel.shape[2]
        square_w = square_kernel.shape[3]
        square_kernel[:, :, square_h // 2 - asym_h // 2:square_h // 2 -
                      asym_h // 2 + asym_h, square_w // 2 - asym_w // 2:
                      square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

    def _pad_tensor(self, tensor, to_size):
        from_size = tensor.shape[-1]
        if from_size == to_size:
            return tensor
        pad = (to_size - from_size) // 2
        return paddle.nn.functional.pad(tensor, [pad, pad, pad, pad])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0

        if isinstance(branch, nn.LayerList):
            fused_kernels = []
            fused_bias = []
            for block in branch:
                kernel = block.conv.weight
                running_mean = block.bn._mean
                running_var = block.bn._variance
                gamma = block.bn.weight
                beta = block.bn.bias
                eps = block.bn._epsilon

                std = (running_var + eps).sqrt()
                t = (gamma / std).reshape((-1, 1, 1, 1))

                fused_kernels.append(kernel * t)
                fused_bias.append(beta - running_mean * gamma / std)

            return sum(fused_kernels), sum(fused_bias)

        elif isinstance(branch, ConvBNLayer):
            kernel = branch.conv.weight
            running_mean = branch.bn._mean
            running_var = branch.bn._variance
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            input_dim = self.in_channels if self.kernel_size == 1 else 1
            kernel_value = paddle.zeros(
                shape=[
                    self.in_channels, input_dim, self.kernel_size,
                    self.kernel_size
                ],
                dtype='float32')
            if self.kernel_size > 1:
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, (self.kernel_size - 1) // 2,
                                 (self.kernel_size - 1) // 2] = 1
            elif self.kernel_size == 1:
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 0, 0] = 1
            else:
                raise ValueError("Invalid kernel size recieved!")
            kernel = paddle.to_tensor(kernel_value, place=branch.weight.place)
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))

        return kernel * t, beta - running_mean * gamma / std
