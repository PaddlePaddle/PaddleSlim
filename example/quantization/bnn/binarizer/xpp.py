import warnings
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Union


class XNORPlusPlusConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=0, transposed=False, output_padding=None, groups=1):
        super(XNORPlusPlusConv2d, self).__init__()
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
        self.o_scale = paddle.create_parameter([1, self.out_channels, 1, 1], dtype="float32", default_initializer=nn.initializer.Assign(paddle.ones([1, self.out_channels, 1, 1])))
        self.h_scale = paddle.create_parameter(shape=[1, 1, 2000, 1], dtype="float32", default_initializer=nn.initializer.Assign(paddle.ones([1, 1, 2000, 1])))
        self.w_scale = paddle.create_parameter(shape=[1, 1, 1, 2000], dtype="float32", default_initializer=nn.initializer.Assign(paddle.ones([1, 1, 1, 2000])))


    def forward(self, x):
        real_input = x
        binary_input_no_grad = paddle.sign(x)
        cliped_input = paddle.clip(x, -1.0, 1.0)
        x = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input

        real_weights = paddle.reshape(self.weight, self.shape)
        binary_weights_no_grad = paddle.sign(real_weights)
        cliped_weights = paddle.clip(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
        o_scale = self.o_scale
        h_scale = self.h_scale[:, :, :y.shape[2], :]
        w_scale = self.w_scale[:, :, :, :y.shape[3]]
        y = y * o_scale * h_scale * w_scale
        return y


class XNORPlusPlusConv1d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=0, transposed=False, output_padding=None, groups=1):
        super(XNORPlusPlusConv1d, self).__init__()
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
        self.weight = nn.Parameter(paddle.rand(*self.shape) * 0.001, requires_grad=True)
        self.o_scale = nn.Parameter(paddle.ones(1, self.out_channels, 1), requires_grad=True)
        self.l_scale = nn.Parameter(paddle.ones(1, 1, 2000), requires_grad=True)


    def forward(self, x):
        real_input = x
        binary_input_no_grad = paddle.sign(x)
        cliped_input = paddle.clip(x, -1.0, 1.0)
        x = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input

        real_weights = paddle.reshape(self.weight, self.shape)
        binary_weights_no_grad = paddle.sign(real_weights)
        cliped_weights = paddle.clip(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv1d(x, binary_weights, stride=self.stride, padding=self.padding)
        o_scale = self.o_scale
        l_scale = self.l_scale[:, :, :y.shape[2]]
        y = y * o_scale * l_scale
        return y

# class BinaryQuantizer(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         out = torch.sign(input)
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         input = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input[input[0].ge(1)] = 0
#         grad_input[input[0].le(-1)] = 0
#         return grad_input


# class XNORPlusPlusLinear(nn.Linear):
#     def __init__(self,  in_features, out_features, bias=True, binary_act=True):
#         super(XNORPlusPlusLinear, self).__init__(in_features, out_features, bias=bias)
#         self.o_scale = nn.Parameter(torch.ones(1, 1, self.weight.shape[0]), requires_grad=True)
 
#     def forward(self, input, type=None):
        
#         ba = BinaryQuantizer.apply(input)
#         bw = BinaryQuantizer.apply(self.weight)
        
#         ba_shape, D = ba.shape[:-1], ba.shape[-1] 
        

#         out = nn.functional.linear(ba.view(-1, D), bw)
#         out = out.view(*ba_shape, -1)

#         if len(out.shape) == 2:
#             out = out * self.o_scale.squeeze(0)
#         else:
#             T = out.shape[1]
#             out = out * self.o_scale
        
#         if not self.bias is None:
#             out += self.bias.view(1, -1).expand_as(out) 

#         return out
