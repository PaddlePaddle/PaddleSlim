import warnings
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Union
import math

class Binary_ReSTE(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input, t, o):
        ctx.save_for_backward(input, t, o)
        out = paddle.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, t, o = ctx.saved_tensor()

        interval = 0.1

        tmp = paddle.zeros_like(input)
        mask1 = (input <= t) & (input > interval)
        tmp[mask1] = (1 / o) * paddle.pow(input[mask1], (1 - o) / o)
        mask2 = (input >= -t) & (input < -interval)
        tmp[mask2] = (1 / o) * paddle.pow(-input[mask2], (1 - o) / o)
        tmp[(input <= interval) & (input >= 0)] = approximate_function(interval, o) / interval
        tmp[(input <= 0) & (input >= -interval)] = -approximate_function(-interval, o) / interval

        # calculate the final gradient
        grad_input = tmp * grad_output.clone()

        return grad_input, None, None

def approximate_function(x, o):
    if x >= 0:
        return math.pow(x, 1 / o)
    else:
        return -math.pow(-x, 1 / o)

class ReSTEConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=0, transposed=False, output_padding=None, groups=1):
        super(ReSTEConv2d, self).__init__()
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
        self.t = paddle.to_tensor(1.5, dtype=paddle.float32, stop_gradient=False)
        self.o = paddle.to_tensor(1., dtype=paddle.float32, stop_gradient=False)
        self.t_a = paddle.to_tensor(1.5, dtype=paddle.float32, stop_gradient=False)
        self.o_a = paddle.to_tensor(1., dtype=paddle.float32, stop_gradient=False)

    def forward(self, x):
        x = Binary_ReSTE.apply(x, self.t_a, self.o_a)

        scaling_factor = paddle.mean(paddle.mean(paddle.mean(paddle.abs(self.weight),axis=3,keepdim=True),axis=2,keepdim=True),axis=1,keepdim=True)
        binary_weights = Binary_ReSTE.apply(self.weight, self.t, self.o) * scaling_factor
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y

