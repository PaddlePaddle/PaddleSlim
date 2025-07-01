import warnings
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Union


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
        # grad_input[paddle.greater_than(input[0], paddle.to_tensor(1.))] = 0
        # grad_input[paddle.less_than(input[0], paddle.to_tensor(-1.))] = 0
        return grad_input


class HOBQConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=0, transposed=False, output_padding=None, groups=1):
        super(HOBQConv2d, self).__init__()
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
        
        self.bias = paddle.create_parameter(shape=[self.out_channels], dtype="float32", default_initializer=nn.initializer.Assign(paddle.rand(shape=[self.out_channels]) * 0.001))
        
        self.scaling_first_order = paddle.create_parameter(shape=[out_channels, 1, 1, 1], dtype="float32", default_initializer=nn.initializer.Assign(paddle.rand([out_channels, 1, 1, 1]) * 0.001))
        self.scaling_second_order = paddle.create_parameter(shape=[out_channels, 1, 1, 1], dtype="float32", default_initializer=nn.initializer.Assign(paddle.rand([out_channels, 1, 1, 1]) * 0.001))
        self.init_scale = False
        
    def forward(self, x):

        bw = self.weight
        if not self.init_scale:
            real_weights = paddle.reshape(self.weight, self.shape)
            scaling_factor = paddle.mean(paddle.mean(paddle.mean(paddle.abs(real_weights),axis=3,keepdim=True),axis=2,keepdim=True),axis=1,keepdim=True)
            self.scaling_first_order = paddle.create_parameter(shape=scaling_factor.shape, dtype="float32", default_initializer=nn.initializer.Assign(scaling_factor))
        
        bw = BinaryQuantize().apply(bw) * self.scaling_first_order    
        first_res_bw = self.weight - bw
        
        if not self.init_scale:
            real_first_res = paddle.reshape(first_res_bw, shape=self.shape)
            scaling_factor = paddle.mean(paddle.mean(paddle.mean(paddle.abs(real_first_res),axis=3,keepdim=True),axis=2,keepdim=True),axis=1,keepdim=True)
            self.scaling_second_order = paddle.create_parameter(shape=scaling_factor.shape, dtype="float32", default_initializer=nn.initializer.Assign(scaling_factor))
            self.init_scale = True
            
        bw = bw + BinaryQuantize().apply(first_res_bw) * self.scaling_second_order
        
        x = BinaryQuantize().apply(x)
        y = F.conv2d(x, bw, stride=self.stride, padding=self.padding, bias=self.bias)

        return y
