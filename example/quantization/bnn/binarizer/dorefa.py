import warnings
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Union


class DoReFaConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=0, transposed=False, output_padding=None, groups=1):
        super(DoReFaConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.number_of_weights = in_channels * out_channels * kernel_size * kernel_size
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = paddle.create_parameter(shape=self.shape, dtype="float32", default_initializer=nn.initializer.Assign(paddle.rand(shape=self.shape) * 0.001))

    def forward(self, x):
        binary_input_no_grad = paddle.sign(x)
        cliped_input = paddle.clip(x, -1.0, 1.0)
        x = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input

        real_weights = paddle.reshape(self.weight, self.shape)
        scaling_factor = paddle.mean(paddle.mean(paddle.mean(paddle.abs(real_weights),axis=3,keepdim=True),axis=2,keepdim=True),axis=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * paddle.sign(real_weights)
        cliped_weights = paddle.clip(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y

