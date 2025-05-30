import warnings
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Union


class BinaryQuantize(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        threshold = paddle.max(paddle.max(paddle.max(paddle.abs(input),axis=3,keepdim=True),axis=2,keepdim=True),axis=1,keepdim=True) * 0.05
        input_tmp = input.clone()
        input_abs = paddle.abs(input_tmp)
        input_tmp[input_tmp.greater_equal(threshold)] = 1
        input_tmp[input_abs.less_equal(threshold)] = 0
        input_tmp[input_tmp.less_equal(-threshold)] = -1
        return input_tmp

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensor()[0]
        input_abs = paddle.abs(input)
        threshold = paddle.max(paddle.max(paddle.max(paddle.abs(input),axis=3,keepdim=True),axis=2,keepdim=True),axis=1,keepdim=True) * 0.05
        grad_input = grad_output.clone()
        grad_input[input.greater_than(threshold)] = input[input.greater_than(threshold)] * grad_output[input.greater_than(threshold)]
        grad_input[input_abs.less_equal(threshold)] = 1 * grad_output[input_abs.less_equal(threshold)]
        grad_input[input.less_equal(-threshold)] = input[input.less_equal(-threshold)] * grad_output[input.less_equal(-threshold)]
        return grad_input

class TTQConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=0, transposed=False, output_padding=None, groups=1):
        super(TTQConv2d, self).__init__()
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
        
        real_weights = paddle.reshape(self.weight, self.shape)
        
        scaling_factor = paddle.mean(paddle.mean(paddle.mean(paddle.abs(real_weights),axis=3,keepdim=True),axis=2,keepdim=True),axis=1,keepdim=True)
        binary_weights = scaling_factor * BinaryQuantize.apply(real_weights)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y
