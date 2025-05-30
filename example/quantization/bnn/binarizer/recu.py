import warnings
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Union
import numpy as np


class ReCUConv2d(nn.Conv2D):

    def __init__(self, *kargs, **kwargs):
        super(ReCUConv2d, self).__init__(*kargs, **kwargs)
        # import pdb; pdb.set_trace()
        self.alpha = paddle.create_parameter(shape=[self.weight.shape[0], 1, 1], dtype="float32", default_initializer=nn.initializer.Assign(paddle.rand([self.weight.shape[0], 1, 1])))
        self.register_buffer('tau', paddle.to_tensor(1.))

    def model_params_update(self, max_epochs, epoch):
        "compute tau"
        a = paddle.to_tensor(np.e)
        T_min, T_max = paddle.to_tensor(0.85).float(), paddle.to_tensor(0.99).float()
        A = (T_max - T_min) / (a - 1)
        B = T_min - A
        tau = A * paddle.to_tensor([paddle.pow(a, epoch / max_epochs)]).float() + B
        self.tau = tau.to(self.tau.device)

    def forward(self, input):
        a = input
        w = self.weight

        w0 = w - paddle.mean(w, axis=[1,2,3], keepdim=True)
        w1 = w0 / (paddle.sqrt(w0.var([1,2,3], keepdim=True) + 1e-5) / 2 / np.sqrt(2))
        EW = paddle.mean(paddle.abs(w1))
        Q_tau = (- EW * paddle.log(2-2*self.tau)).detach().cpu().item()
        w2 = paddle.clip(w1, -Q_tau, Q_tau)

        if self.training:
            a0 = a / paddle.sqrt(a.var([1,2,3], keepdim=True) + 1e-5)
        else: 
            a0 = a
        
        #* binarize
        bw = BinaryQuantize().apply(w2)
        ba = BinaryQuantize_a().apply(a0)
        #* 1bit conv
        output = F.conv2d(ba, bw, self.bias,
                          self._stride, self._padding,
                          self._dilation, self._groups)
        #* scaling factor
        output = output * self.alpha
        return output

class BinaryQuantize(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input):
        out = paddle.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryQuantize_a(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = paddle.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensor()[0]
        grad_input = (2 - paddle.abs(2*input))
        grad_input = grad_input.clip(min=0) * grad_output.clone()
        return grad_input


class BinaryQuantizeW(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input):
        out = paddle.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryQuantizeA(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = paddle.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensor()[0]
        grad_input = (2 - paddle.abs(2*input))
        grad_input = grad_input.clip(min=0) * grad_output.clone()
        return grad_input
