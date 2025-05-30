import warnings
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Union
import numpy as np


class FDA_BinaryQuantize(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, inputs, n):
        ctx.save_for_backward(inputs, n)
        out = paddle.sign(inputs)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, n = ctx.saved_tensor()
        omega = 0.1
        grad_input = 4 * omega / np.pi * sum(([paddle.cos((2 * i + 1) * omega * inputs) for i in range(n + 1)])) * grad_output
        grad_input[paddle.greater_than(inputs, paddle.to_tensor(1.))] = 0
        grad_input[paddle.less_than(inputs, paddle.to_tensor(-1.))] = 0
        return grad_input, None


class NoiseAdaption(nn.Layer):

    def __init__(self, d, k=64):
        super(NoiseAdaption, self).__init__()
        self.fc1 = nn.Linear(in_features=d, out_features=d // k, bias_attr=False)
        self.fc2 = nn.Linear(in_features=d // k, out_features=d, bias_attr=False)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        shape = x.shape
        _x = x
        x = paddle.reshape(x, shape=[shape[0],-1])
        x = self.fc1(x)
        x = F.relu(x)
        x = paddle.reshape(self.fc2(x), shape=shape)
        x += 0.1 * paddle.sin(_x)
        return x


class NoiseAdaption_a(nn.Layer):

    def __init__(self, d, k=64):
        super(NoiseAdaption_a, self).__init__()
        self.fc1 = nn.Linear(in_features=d, out_features=d // k, bias_attr=False)
        self.fc2 = nn.Linear(in_features=d // k, out_features=d, bias_attr=False)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        shape = x.shape
        _x = x
        x = paddle.reshape(x, shape=[shape[0],shape[1],-1])
        x = x.transpose([0, 2, 1])

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.transpose([0, 2, 1]).reshape(shape)
        x += 0.1 * paddle.sin(_x)
        return x


class FDAConv2d(nn.Conv2D):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, noise_adapt=True):
        super(FDAConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias_attr=bias)
        self.register_buffer('n', paddle.to_tensor(9))
        self.register_buffer('alpha', paddle.to_tensor(0.1))
        self.noise_adapt = noise_adapt
        d, k = int(paddle.prod(paddle.to_tensor(self.weight.shape[1:])).numpy()), 64
        if self.noise_adapt:
            self.act_noise = None
            try:
                self.weight_noise = NoiseAdaption(d=d, k=64)
            except:
                self.weight_noise = NoiseAdaption(d=d, k=4)

    def model_params_update(self, max_epochs, epoch):
        alpha = 0.05 * (1 +paddle.cos(paddle.to_tensor((epoch + 1) / max_epochs) * np.pi))
        self.alpha = alpha.to(self.alpha)
        self.n = paddle.to_tensor(9 + int(epoch / max_epochs * 9))

    def forward(self, input):
        w = self.weight
        a = input
        bw = FDA_BinaryQuantize().apply(w, self.n)
        ba = FDA_BinaryQuantize().apply(a, self.n)

        if self.noise_adapt:
            if self.act_noise is None:
                d = int(paddle.prod(paddle.to_tensor(input.shape[1])).numpy())
                try:
                    self.act_noise = NoiseAdaption_a(d=d, k=64)
                except:
                    self.act_noise = NoiseAdaption_a(d=d, k=4)

            ew = self.weight_noise(w) 
            bw = bw + self.alpha * ew

            ea = self.act_noise(a)
            ba = ba + self.alpha * ea
       
        scaling_factor = paddle.mean(paddle.mean(paddle.mean(paddle.abs(self.weight),axis=3,keepdim=True),axis=2,keepdim=True),axis=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        bw = bw * scaling_factor
        output = F.conv2d(ba, bw, self.bias,
                          self._stride, self._padding,
                          self._dilation, self._groups)
        return output
