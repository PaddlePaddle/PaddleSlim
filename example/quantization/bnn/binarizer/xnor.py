import warnings
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Union


class XNORConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=0, transposed=False, output_padding=None, groups=1):
        super(XNORConv2d, self).__init__()
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
        real_input = x
        scaling_factor_x = paddle.mean(paddle.mean(paddle.mean(paddle.abs(real_input),axis=3,keepdim=True),axis=2,keepdim=True),axis=1,keepdim=True)
        scaling_factor_x = scaling_factor_x.detach()
        binary_input_no_grad = scaling_factor_x * paddle.sign(x)
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


class XNORConv1d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=0, transposed=False, output_padding=None, groups=1):
        super(XNORConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.number_of_weights = in_channels * out_channels * kernel_size
        self.shape = (out_channels, in_channels, kernel_size)
        self.weight = paddle.create_parameter(shape=self.shape, dtype="float32", default_initializer=nn.initializer.Assign(paddle.rand(shape=self.shape) * 0.001))

    def forward(self, x):
        real_input = x
        scaling_factor_x = paddle.mean(paddle.mean(paddle.abs(real_input),axis=2,keepdim=True),axis=1,keepdim=True)
        scaling_factor_x = scaling_factor_x.detach()
        binary_input_no_grad = scaling_factor_x * paddle.sign(x)
        cliped_input = paddle.clip(x, -1.0, 1.0)
        x = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input

        real_weights = self.weight.view(self.shape)
        scaling_factor = paddle.mean(paddle.mean(paddle.abs(real_weights),axis=2,keepdim=True),axis=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * paddle.sign(real_weights)
        cliped_weights = paddle.clip(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv1d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y


class BinaryQuantize(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = paddle.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensor()
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input


class XNORLinear(paddle.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(XNORLinear, self).__init__(in_features, out_features, bias=bias)
        self.binary_act = binary_act

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = bw - paddle.reshape(paddle.mean(bw, axis=-1), shape=[-1, 1])
        sw = paddle.reshape(paddle.mean(paddle.abs(bw), axis=-1), shape=[-1, 1]).detach()
        bw = BinaryQuantize().apply(bw)
        bw = bw * sw
        if self.binary_act:
            sa = paddle.reshape(paddle.mean(paddle.abs(ba), axis=-1), shape=[-1, 1]).detach()
            ba = BinaryQuantize().apply(ba)
            ba = ba * sa
        output = F.linear(ba, bw, self.bias)
        return output
