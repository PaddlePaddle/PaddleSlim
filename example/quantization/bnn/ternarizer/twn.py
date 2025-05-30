import warnings
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from torch.autograd import Function

class Ternary(Function):

    @staticmethod
    def forward(self, input):
        E = paddle.mean(paddle.mean(paddle.mean(paddle.abs(input),axis=3,keepdim=True),axis=2,keepdim=True),axis=1,keepdim=True)
        threshold = E * 0.7
        output = paddle.sign(paddle.add(paddle.sign(paddle.add(input, threshold)),paddle.sign(paddle.add(input, -threshold))))
        return output, threshold

    @staticmethod
    def backward(self, grad_output, grad_threshold):
        grad_input = grad_output.clone()
        return grad_input


class TWNConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=0, transposed=False, output_padding=None, groups=1):
        super(TWNConv2d, self).__init__()
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
        self.W = 3

    def forward(self, x):
        if self.W == 3:
            output_fp = self.weight.clone()
            output, threshold = Ternary.apply(self.weight)
            output_abs = paddle.abs(output_fp)
            mask_le = output_abs.less_equal(threshold)
            mask_gt = output_abs.greater_than(threshold)
            output_abs[mask_le] = 0
            output_abs_th = output_abs.clone()
            output_abs_th_sum = paddle.sum(output_abs_th, (3, 2, 1), keepdim=True)
            mask_gt_sum = paddle.sum(mask_gt, (3, 2, 1), keepdim=True).cast('float32')
            alpha = output_abs_th_sum / mask_gt_sum
            ternary_weights = output * alpha
        else:
            ternary_weights = self.weight
        y = F.conv2d(x, ternary_weights, stride=self.stride, padding=self.padding)
        return y
