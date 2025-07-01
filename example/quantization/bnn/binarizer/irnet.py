import paddle.nn as nn
import paddle.nn.functional as F
import paddle
import math


class BinaryQuantize(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = paddle.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensor()
        grad_input = k * t * (1 - paddle.pow(paddle.tanh(input * t), 2)) * grad_output
        return grad_input, None, None


class IRConv2d(nn.Conv2D):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias_attr=bias)
        self.k = paddle.to_tensor([10]).cast('float32')
        self.t = paddle.to_tensor([0.1]).cast('float32')

    def forward(self, input):
        w = self.weight
        a = input
        bw = w - w.reshape([w.shape[0], -1]).mean(-1).reshape([w.shape[0], 1, 1, 1])
        bw = bw / bw.reshape([bw.shape[0], -1]).std(-1).reshape([bw.shape[0], 1, 1, 1])
        sw = paddle.pow(paddle.to_tensor([2]*bw.shape[0]).cast('float32'), (paddle.log(bw.abs().reshape([bw.shape[0], -1]).mean(-1)) / math.log(2)).round().cast('float32')).reshape([bw.shape[0], 1, 1, 1]).detach()
        bw = BinaryQuantize().apply(bw, self.k, self.t)
        ba = BinaryQuantize().apply(a, self.k, self.t)
        bw = bw * sw
        output = F.conv2d(ba, bw, self.bias,
                          self._stride, self._padding,
                          self._dilation, self._groups)
        return output
