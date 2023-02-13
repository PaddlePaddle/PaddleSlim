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

__all__ = ["RepVGGBlock"]


class RepVGGBlock(BaseConv2DReper):
    """
    An instance of the RepVGGBlock module, which replaces the conv-bn layer in the network.
    Refer from Paper: https://arxiv.org/abs/2101.03697.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 padding=None):
        super(RepVGGBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding)
        # Re-parameterizable skip connection
        self.rbr_skip = nn.BatchNorm2D(
            num_features=in_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0))
        ) if in_channels == out_channels and self.stride == 1 else None

        # Re-parameterizable conv branches
        self.rbr_conv = ConvBNLayer(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            groups=self.groups)

        # Re-parameterizable scale branch
        self.rbr_scale = None
        if kernel_size > 1:
            self.rbr_scale = ConvBNLayer(
                self.in_channels,
                self.out_channels,
                1,
                stride=self.stride,
                groups=self.groups)

    def forward(self, x):
        # Inference mode forward pass.
        if hasattr(self, "reparam_conv"):
            return self.reparam_conv(x)

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        out += self.rbr_conv(x)

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
        if hasattr(self, 'rbr_scale'):
            self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

    def _get_kernel_bias(self):
        """ 
        Method to obtain re-parameterized kernel and bias.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size. 1x1->3x3
            padding_size = self.kernel_size // 2
            kernel_scale = paddle.nn.functional.pad(kernel_scale, [
                padding_size, padding_size, padding_size, padding_size
            ])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv, bias_conv = self._fuse_bn_tensor(self.rbr_conv)

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

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
