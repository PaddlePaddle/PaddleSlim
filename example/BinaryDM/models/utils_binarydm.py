import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import math

def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1D(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2D(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3D(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class BNNConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=0, transposed=False, output_padding=None, groups=1, precision='bnn', order=2):
        super(BNNConv2d, self).__init__()
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
        self.shape = [out_channels, in_channels, kernel_size, kernel_size]
        # tmp = paddle.rand(self.shape) * 0.001
        self.weight = paddle.create_parameter(shape=self.shape, dtype=paddle.float32)
        # tmp = paddle.rand(out_channels) * 0.001
        self.bias = paddle.create_parameter(shape=[out_channels], dtype=paddle.float32)
        
        self.order = order
        self.scaling_first_order = paddle.create_parameter(shape=[out_channels, 1, 1, 1], dtype=paddle.float32)
        self.scaling_second_order = paddle.create_parameter(shape=[out_channels, 1, 1, 1], dtype=paddle.float32)
        # paddle.create_parameter(shape=tmp.shape, dtype=tmp.dtype, default_initializer=nn.initializer.Assign(tmp))
        # self.sw = None
        self.init_scale = False
        
        self.precision = precision
        self.bnn_mode = 'bnn'

        self.binary_act = False
        
        self.is_int = False
        # tmp = paddle.rand([out_channels, 1, 1, 1]) * 0.001
        self.nbits = 8
        self.Qn = -2 ** (self.nbits - 1)
        self.Qp = 2 ** (self.nbits - 1) - 1
        self.n_levels = 2 ** self.nbits
        
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2D(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, padding=0)
        tmp = paddle.ones([1]) * 0.3
        self.shortcut_scale = paddle.create_parameter(shape=tmp.shape, dtype=tmp.dtype, default_initializer=nn.initializer.Assign(tmp))

    def forward(self, x, bnn_mode='bnn'):
        
        x_raw = x
        
        if 'full' in [self.precision, self.bnn_mode, bnn_mode]:
            return F.conv2d(x, self.weight, stride=self.stride, padding=self.padding, bias=self.bias)

        bw = self.weight
        if not self.init_scale:
            real_weights = self.weight.reshape(self.shape)
            scaling_factor = paddle.mean(paddle.mean(paddle.mean(abs(real_weights),axis=3,keepdim=True),axis=2,keepdim=True),axis=1,keepdim=True)
            self.scaling_first_order = paddle.create_parameter(shape=scaling_factor.shape, dtype=scaling_factor.dtype, default_initializer=nn.initializer.Assign(scaling_factor))
        
        bw_fp = bw * self.scaling_first_order
        bw = (paddle.sign(bw) * self.scaling_first_order).detach() - bw_fp.detach() + bw_fp
        
        if self.order == 1:
            y = F.conv2d(x, bw, stride=self.stride, padding=self.padding, bias=self.bias)
            if self.in_channels == self.out_channels:
                if x_raw.shape[-1] < y.shape[-1]:
                    shortcut = F.interpolate(x_raw, scale_factor=2, mode="nearest")
                elif x_raw.shape[-1] > y.shape[-1]:
                    shortcut = avg_pool_nd(2, kernel_size=self.stride, stride=self.stride)(x_raw)
                else:
                    shortcut = x_raw
            else:
                shortcut = self.shortcut(x_raw)
            return y + shortcut * paddle.abs(self.shortcut_scale)

        first_res_bw = self.weight - bw
        
        if not self.init_scale:
            real_first_res = first_res_bw.view(self.shape)
            scaling_factor = paddle.create_parameter(shape=real_first_res.shape, dtype=real_first_res.dtype, default_initializer=nn.initializer.Assign(real_first_res))
            self.scaling_second_order.data = scaling_factor
            self.init_scale = True
            
        bw_fp = first_res_bw * self.scaling_second_order
        bw = (paddle.sign(first_res_bw) * self.scaling_second_order).detach() - bw_fp.detach() + bw_fp
        
        y = F.conv2d(x, bw, stride=self.stride, padding=self.padding, bias=self.bias)

        if self.in_channels == self.out_channels:
            if x_raw.shape[-1] < y.shape[-1]:
                shortcut = F.interpolate(x_raw, scale_factor=2, mode="nearest")
            elif x_raw.shape[-1] > y.shape[-1]:
                shortcut = avg_pool_nd(2, kernel_size=self.stride, stride=self.stride)(x_raw)
            else:
                shortcut = x_raw
        else:
            shortcut = self.shortcut(x_raw)
        return y + shortcut * paddle.abs(self.shortcut_scale)
