# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
"""
Wrapper layer to simulate folding batch norms during quantization-aware training.
"""
import paddle
from paddle.nn import Layer
from paddle.nn import functional as F
from paddle.nn.layer.norm import _BatchNormBase
from paddle.nn.quant.format import ConvertibleQuantedLayer


class BatchNorm(Layer):
    r"""Wrapper of batchnorm layer. It is used to get the mean and variance of cerrent mini-batch.
    Args:
        - bn(paddle.nn.layer.norm._BatchNormBase):  The batchnorm layer to be wrapped.
    """

    def __init__(self, bn: _BatchNormBase):
        super(BatchNorm, self).__init__()
        self._bn = bn
        self._runing_mean = self._bn._mean
        self._runing_variance = self._bn._variance
        self._weight = self._bn.weight
        self._bias = self._bn.bias
        self._batch_mean = None
        self._batch_variance = None
        self._training = True
        self._batch_mean = None
        self._batch_variance = None
        assert self._bias is not None

    def forward(self, input):
        r"""
        Args:
        - input(Tensor): The input to be normalized.
        Return:
        A tuple with 3 elements that is normalized output, mean of current mini-natch and variance of
        current mini-batch.
        """
        (
            batch_norm_out, _, _, batch_mean, batch_variance, _,
        ) = paddle._C_ops.batch_norm(
            input,
            self.
            _runing_mean,  # TODO: should freeze this runing mean to avoid instability in training
            self.
            _runing_variance,  # TODO: should freeze this runing variance to avoid instability in training
            self._weight,
            self._bias,
            not self._training,
            self._bn._momentum,
            self._bn._epsilon,
            self._bn._data_format,
            False,  # use_global_stats
            False,  # trainable_statistics
        )
        batch_mean.stop_gradient = True
        batch_variance.stop_gradient = True
        self._batch_mean = batch_mean
        self._batch_variance = batch_variance
        return batch_norm_out, batch_mean, batch_variance


class Conv2DBatchNormWrapper(paddle.nn.Layer):
    def __init__(self, conv, bn):
        super(Conv2DBatchNormWrapper, self).__init__()
        self._conv = conv
        self._bn = bn
        self._freeze_bn_delay = None

    def forward(self, inputs):
        return self._bn(self._conv(inputs))


class QuantedConv2DBatchNorm(ConvertibleQuantedLayer):
    r"""Wrapper layer to simulate folding batch norms during quantization-aware training.
    Fisrtly, it will execute once convolution and batch norms prior to quantizing weights
    to get the long term statistics(i.e. runing mean and runing variance).
    And it always scale the convolution's weights with a correction factor to the long term
    statistics prior to quantization. This ensures that there is no jitter in the quantized
    weights due to batch to batch variation.
    Then the training will be divided into two phases:
    1. During the initial phase of training, with freeze_bn is false. It undo the scaling of
    the weights so that outputs are identical to regular batch normalization.
    2. After sufficient training， with freeze_bn is true. Switch from using batch statistics
    to long term moving averages for batch normalization.
    Reference: Quantizing deep convolutional networks for efficient inference: A whitepaper
    """

    def __init__(self, conv_bn: Conv2DBatchNormWrapper, q_config):
        super(QuantedConv2DBatchNorm, self).__init__()
        conv = conv_bn._conv
        bn = conv_bn._bn
        self._freeze_bn_delay = conv_bn._freeze_bn_delay
        # For Conv2D
        self._groups = getattr(conv, '_groups')
        self._stride = getattr(conv, '_stride')
        self._padding = getattr(conv, '_padding')
        self._padding_mode = getattr(conv, '_padding_mode')
        if self._padding_mode != 'zeros':
            self._reversed_padding_repeated_twice = getattr(
                conv, '_reversed_padding_repeated_twice')
        self._dilation = getattr(conv, '_dilation')
        self._data_format = getattr(conv, '_data_format')
        self.conv_weight = getattr(conv, 'weight')
        self.conv_bias = getattr(conv, 'bias')

        if self.conv_bias is None:
            self.conv_bias = paddle.create_parameter(
                bn.bias.shape, dtype=self.conv_weight.dtype, is_bias=True)

        self.bn = BatchNorm(bn)

        self.weight_quanter = None
        self.activation_quanter = None
        if q_config.weight is not None:
            self.weight_quanter = q_config.weight._instance(conv)
        if q_config.activation is not None:
            self.activation_quanter = q_config.activation._instance(conv)

        self._freeze_bn = False

        self._steps = 0

    def freeze_bn(self):
        self._freeze_bn = True

    def unfreeze_bn(self):
        self._freeze_bn = False

    def forward(self, input):
        if self.training:
            if self._freeze_bn_delay == self._steps:
                self.freeze_bn()

            if self._freeze_bn:
                out = self._forward_with_bn_freezed(input)
            else:
                out = self._forward_with_bn_unfreezed(input)
            self._steps += 1
            return out
        else:
            return self._eval_forward(input)

    def _forward_with_bn_unfreezed(self, input):
        quant_input = input
        if self.activation_quanter is not None:
            quant_input = self.activation_quanter(quant_input)

        # Step 1: Excute conv bn to get global runing mean and variance
        _, _, batch_variance = self._conv_bn(quant_input, self.conv_weight)
        # Step 2: Merge and quantize weights of conv and bn
        merged_weight = self._merge_conv_bn_weight(self.conv_weight, self.bn)
        quant_weight = merged_weight
        if self.weight_quanter is not None:
            quant_weight = self.weight_quanter(quant_weight)
        # Step 3: Excute conv with merged weights
        conv_out = self._conv_forward(quant_input, quant_weight)
        # Step 4: Scale output of conv and merge bias
        factor = paddle.reshape((self.bn._runing_variance / batch_variance),
                                [1, -1, 1, 1])
        factor.stop_gradient = True
        conv_out = conv_out * factor
        merged_bias = self._merge_conv_unfreezed_bn_bias(
            self.conv_bias, self.bn)
        return conv_out + paddle.reshape(merged_bias, [1, -1, 1, 1])

    def _eval_forward(self, input):
        quant_input = input
        if self.activation_quanter is not None:
            quant_input = self.activation_quanter(quant_input)
        if self.weight_quanter is not None:
            quant_weight = self.weight_quanter(self.conv_weight)
        conv_out = self._conv_forward(quant_input, quant_weight)
        conv_out = conv_out if self.conv_bias is None else conv_out + paddle.reshape(
            self.conv_bias, [1, -1, 1, 1])
        return conv_out

    def _forward_with_bn_freezed(self, input):
        quant_input = input
        if self.activation_quanter is not None:
            quant_input = self.activation_quanter(quant_input)
        # Step 1: Excute conv bn to get global runing mean and variance
        self._conv_bn(quant_input, self.conv_weight)
        # Step 2: Merge and quantize weights of conv and bn
        merged_weight = self._merge_conv_bn_weight(self.conv_weight, self.bn)
        quant_weight = merged_weight
        if self.weight_quanter is not None:
            quant_weight = self.weight_quanter(quant_weight)
        # Step 3: Excute conv with merged weights
        conv_out = self._conv_forward(quant_input, quant_weight)
        # Step 4: Merge bias of conv and bn
        merged_bias = self._merge_conv_freezed_bn_bias(self.conv_bias, self.bn)
        return conv_out + paddle.reshape(merged_bias, [1, -1, 1, 1])

    def _merge_conv_bn_weight(self, conv_weight, bn: BatchNorm):
        merged_weight = paddle.reshape(
            bn._weight, [-1, 1, 1, 1]) * conv_weight / paddle.reshape(
                bn._runing_variance, [-1, 1, 1, 1])
        conv_weight.set_value(merged_weight)
        return conv_weight

    def _merge_conv_freezed_bn_bias(self, conv_bias, bn: BatchNorm):
        assert conv_bias is not None

        merged_bias = (conv_bias + bn._bias -
                       (bn._weight * bn._runing_mean) / bn._runing_variance)
        conv_bias.set_value(merged_bias)
        return conv_bias

    def _merge_conv_unfreezed_bn_bias(self, conv_bias, bn: BatchNorm):
        assert conv_bias is not None
        merged_bias = (conv_bias + bn._bias -
                       bn._weight * bn._batch_mean / bn._batch_variance)
        conv_bias.set_value(merged_bias)
        return conv_bias

    def _conv_bn(self, input, conv_weight):
        return self._bn_forward(self._conv_forward(input, conv_weight))

    def _bn_forward(self, input):
        return self.bn.forward(input)

    def _conv_forward(self, inputs, weights):
        if self._padding_mode != 'zeros':
            inputs = F.pad(
                inputs,
                self._reversed_padding_repeated_twice,
                mode=self._padding_mode,
                data_format=self._data_format, )
            self._padding = 0

        return F.conv2d(
            inputs,
            weights,
            bias=self.conv_bias,
            padding=self._padding,
            stride=self._stride,
            dilation=self._dilation,
            groups=self._groups,
            data_format=self._data_format, )

    def weights_to_quanters(self):
        return [('conv_weight', 'weight_quanter')]

    def activation_quanters(self):
        return ['activation_quanter']
