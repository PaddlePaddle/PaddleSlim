import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .ops import forward_quantizer, quantized_linear_for_training, quantized_conv2d_for_training
from .quantization_utils import parse_qtype
from . import quantizer

# 定义 _pair 函数
# def _pair(x):
#     if isinstance(x, (tuple, list)):
#         return x
#     return (x, x)
import collections
from itertools import repeat
def _pair(x):
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, 2))


'''
Quantization-aware Training
'''
class QModule(nn.Layer):
    def __init__(self):
        super(QModule, self).__init__()

        self.calibrating = False
        self.input_scale_list = []

        # input quantization info
        self.input_qtype = None
        self.input_granularity = None
        self.register_buffer('input_bitwidth', None)
        self.register_buffer('input_scale', None)
        self.input_quantizer = None

        # weight quantization info
        self.weight_qtype = None
        self.weight_granularity = None
        self.register_buffer('weight_bitwidth', None)
        self.register_buffer('weight_scale', None)
        self.weight_quantizer = None

    def init(self, qparam):
        # input quantizer
        device = self.weight.place
        qtype_name, bitwidth = parse_qtype(qparam['input_qtype'])
        self.input_qtype = qtype_name
        self.input_granularity = qparam['input_granularity']
        self.register_buffer('input_bitwidth', paddle.to_tensor(bitwidth).astype('int32').place(device))
        self.register_buffer('input_scale', paddle.to_tensor(1).astype('float32').place(device))
        self.input_quantizer = quantizer.__dict__[qtype_name + "_quantizer"](bitwidth, self.input_granularity, axis=1)

        # weight quantizer
        qtype_name, bitwidth = parse_qtype(qparam['weight_qtype'])
        self.weight_qtype = qtype_name
        self.weight_granularity = qparam['weight_granularity']
        self.register_buffer('weight_bitwidth', paddle.to_tensor(bitwidth).astype('int32').place(device))
        self.register_buffer('weight_scale', paddle.to_tensor(1).astype('float32').place(device))
        self.weight_quantizer = quantizer.__dict__[qtype_name + "_quantizer"](bitwidth, self.weight_granularity)

    def __setattr__(self, name, value):
        if name in ('input_bitwidth', 'input_scale', 'weight_bitwidth', 'weight_scale'):
            device = getattr(self, name).place
            if name == 'input_bitwidth':
                if isinstance(value, paddle.Tensor):
                    del self._buffers['input_bitwidth']
                    self._buffers['input_bitwidth'] = value.place(device)
                elif isinstance(value, int):
                    del self._buffers['input_bitwidth']
                    self._buffers['input_bitwidth'] = paddle.to_tensor(value).astype('int32').place(device)
                else:
                    raise TypeError
            elif name == 'input_scale':
                if isinstance(value, paddle.Tensor):
                    del self._buffers['input_scale']
                    self._buffers['input_scale'] = value.place(device)
                elif isinstance(value, int) or isinstance(value, float):
                    del self._buffers['input_scale']
                    self._buffers['input_scale'] = paddle.to_tensor(value).astype('float32').place(device)
                else:
                    raise TypeError
            if name == 'weight_bitwidth':
                if isinstance(value, paddle.Tensor):
                    del self._buffers['weight_bitwidth']
                    self._buffers['weight_bitwidth'] = value.place(device)
                elif isinstance(value, int):
                    del self._buffers['weight_bitwidth']
                    self._buffers['weight_bitwidth'] = paddle.to_tensor(value).astype('int32').place(device)
                else:
                    raise TypeError
            elif name == 'weight_scale':
                if isinstance(value, paddle.Tensor):
                    del self._buffers['weight_scale']
                    self._buffers['weight_scale'] = value.place(device)
                elif isinstance(value, int) or isinstance(value, float):
                    del self._buffers['weight_scale']
                    self._buffers['weight_scale'] = paddle.to_tensor(value).astype('float32').place(device)
                else:
                    raise TypeError
        else:
            super(QModule, self).__setattr__(name, value)

    def initialize_quantizer(self):
        self.input_scale = paddle.stack(self.input_scale_list).mean(axis=0)
        self.input_scale_list = []
        self.input_quantizer.set_scale(self.input_scale)

        weight_scale = self.weight_quantizer.calculate_quantization_scale(self.weight.numpy())
        self.weight_scale = weight_scale
        self.weight_quantizer.set_scale(self.weight_scale)

    def forward(self, input):
        pass


class QLinear(nn.Linear, QModule):
    def __init__(self, input_features, output_features, bias=True):
        super(QLinear, self).__init__(input_features, output_features, bias)

    def forward(self, input):
        if self.calibrating:
            scale = self.input_quantizer.calculate_quantization_scale(input.numpy())
            self.input_scale_list.append(scale)
            return super(QLinear, self).forward(input)
        else:
            input_q = forward_quantizer.apply(input, self.input_quantizer)
            weight_q = forward_quantizer.apply(self.weight, self.weight_quantizer)

            return paddle.nn.functional.linear(input_q, weight_q, self.bias)


class QConv2d(nn.Conv2D, QModule):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, input):
        if self.calibrating:
            scale = self.input_quantizer.calculate_quantization_scale(input.detach())
            self.input_scale_list.append(scale)
            return super(QConv2d, self).forward(input)
        else:
            input_q = forward_quantizer.apply(input, self.input_quantizer)
            weight_q = forward_quantizer.apply(self.weight, self.weight_quantizer)

            if self._padding_mode != 'zeros':
                return paddle.nn.functional.conv2d(F.pad(input_q, self._reversed_padding_repeated_twice, padding_mode=self._padding_mode),
                                weight_q, self.bias, self._stride,
                                _pair(0), self._dilation, self._groups)
            return paddle.nn.functional.conv2d(input_q, weight_q, self.bias, self._stride,
                            self._padding, self._dilation, self._groups)



'''
Training Quantization
'''
class QTModule(nn.Layer):
    def __init__(self):
        super(QTModule, self).__init__()

        self.enabled = False
        # weight quantization info
        self._hybrid_weight = False
        self._hybrid_input = False
        self._hybrid_grad = False

    def init(self, qparam):
        self.enabled = True
        device = self.weight.place

        # weight quantizer
        if qparam.weight_forward == qparam.weight_backward:
            # single quantizer
            self._hybrid_weight = False
            qtype_name, bitwidth = parse_qtype(qparam.weight_forward.qtype)
            self.weight_forward_quantizer = quantizer.__dict__[qtype_name + "_quantizer"](bitwidth, signed=True, 
                                                            scaling=qparam.weight_forward.scaling,
                                                            dims=qparam.weight_forward.granularity,
                                                            observer=qparam.weight_forward.observer,
                                                            stochastic=qparam.weight_forward.stochastic)
            self.weight_backward_quantizer = self.weight_forward_quantizer
        else:
            # hybrid quantizer
            self._hybrid_weight = True
            qtype_name, bitwidth = parse_qtype(qparam.weight_forward.qtype)
            self.weight_forward_quantizer = quantizer.__dict__[qtype_name + "_quantizer"](bitwidth, signed=True, 
                                                            scaling=qparam.weight_forward.scaling,
                                                            dims=qparam.weight_forward.granularity,
                                                            observer=qparam.weight_forward.observer,
                                                            stochastic=qparam.weight_forward.stochastic)
            qtype_name, bitwidth = parse_qtype(qparam.weight_backward.qtype)
            self.weight_backward_quantizer = quantizer.__dict__[qtype_name + "_quantizer"](bitwidth, signed=True, 
                                                            scaling=qparam.weight_backward.scaling,
                                                            dims=qparam.weight_backward.granularity,
                                                            observer=qparam.weight_backward.observer,
                                                            stochastic=qparam.weight_backward.stochastic)

        # input quantizer
        if qparam.input_forward == qparam.input_backward:
            # single quantizer
            self._hybrid_input = False
            qtype_name, bitwidth = parse_qtype(qparam.input_forward.qtype)
            self.input_forward_quantizer = quantizer.__dict__[qtype_name + "_quantizer"](bitwidth, 
                                                            scaling=qparam.input_forward.scaling,
                                                            dims=qparam.input_forward.granularity,
                                                            observer=qparam.input_forward.observer,
                                                            stochastic=qparam.input_forward.stochastic)
            self.input_backward_quantizer = self.input_forward_quantizer
        else:
            # hybrid quantizer
            self._hybrid_input = True
            qtype_name, bitwidth = parse_qtype(qparam.input_forward.qtype)
            self.input_forward_quantizer = quantizer.__dict__[qtype_name + "_quantizer"](bitwidth, 
                                                            scaling=qparam.input_forward.scaling,
                                                            dims=qparam.input_forward.granularity,
                                                            observer=qparam.input_forward.observer,
                                                            stochastic=qparam.input_forward.stochastic)
            qtype_name, bitwidth = parse_qtype(qparam.input_backward.qtype)
            self.input_backward_quantizer = quantizer.__dict__[qtype_name + "_quantizer"](bitwidth, 
                                                            scaling=qparam.input_backward.scaling,
                                                            dims=qparam.input_backward.granularity,
                                                            observer=qparam.input_backward.observer,
                                                            stochastic=qparam.input_backward.stochastic)

        # grad quantizer
        if qparam.grad_input == qparam.grad_weight:
            # single quantizer
            self._hybrid_grad = False
            qtype_name, bitwidth = parse_qtype(qparam.grad_input.qtype)
            self.grad_input_quantizer = quantizer.__dict__[qtype_name + "_quantizer"](bitwidth, signed=True, 
                                                            scaling=qparam.grad_input.scaling,
                                                            dims=qparam.grad_input.granularity,
                                                            observer=qparam.grad_input.observer,
                                                            stochastic=qparam.grad_input.stochastic)
            self.grad_weight_quantizer = self.grad_input_quantizer
        else:
            # hybrid quantizer
            self._hybrid_grad = True
            qtype_name, bitwidth = parse_qtype(qparam.grad_input.qtype)
            self.grad_input_quantizer = quantizer.__dict__[qtype_name + "_quantizer"](bitwidth, signed=True, 
                                                            scaling=qparam.grad_input.scaling,
                                                            dims=qparam.grad_input.granularity,
                                                            observer=qparam.grad_input.observer,
                                                            stochastic=qparam.grad_input.stochastic)
            qtype_name, bitwidth = parse_qtype(qparam.grad_weight.qtype)
            self.grad_weight_quantizer = quantizer.__dict__[qtype_name + "_quantizer"](bitwidth, signed=True, 
                                                            scaling=qparam.grad_weight.scaling,
                                                            dims=qparam.grad_weight.granularity,
                                                            observer=qparam.grad_weight.observer,
                                                            stochastic=qparam.grad_weight.stochastic)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


class QTLinear(nn.Linear, QTModule):
    def __init__(self, input_features, output_features, bias=True):
        nn.Linear.__init__(self, input_features, output_features, bias)
        QTModule.__init__(self)

    def forward(self, input):
        if self.enabled:
            # print(input.shape)  # [256, 512]
            # print(self.weight.shape)  # [512, 1000] 这里torch是 [1000, 512]
            # assert 0==1
            return quantized_linear_for_training.apply(input, self.weight, self.bias, 
                                        self._hybrid_weight, self.weight_forward_quantizer, self.weight_backward_quantizer,
                                        self._hybrid_input, self.input_forward_quantizer, self.input_backward_quantizer,
                                        self._hybrid_grad, self.grad_input_quantizer, self.grad_weight_quantizer)
        else:
            if self.training:
                print("Warning: quantization is disabled!!!")
            return super(QTLinear, self).forward(input)


class QTConv2d(nn.Conv2D, QTModule):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        nn.Conv2D.__init__(self, in_channels, out_channels, kernel_size,
                           stride=stride, padding=padding, dilation=dilation, groups=groups, 
                           bias_attr=bias, padding_mode=padding_mode)
        QTModule.__init__(self)
        


    def forward(self, input):
        if self.enabled:
            if self._padding_mode != 'zeros':
                return quantized_conv2d_for_training.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self._padding_mode),
                                            self.weight, self.bias, self._stride, _pair(0), self._dilation, self._groups, 
                                            self._hybrid_weight, self.weight_forward_quantizer, self.weight_backward_quantizer,
                                            self._hybrid_input, self.input_forward_quantizer, self.input_backward_quantizer,
                                            self._hybrid_grad, self.grad_input_quantizer, self.grad_weight_quantizer)
            return quantized_conv2d_for_training.apply(input, self.weight, self.bias, self._stride,
                                                self._padding, self._dilation, self._groups, 
                                                self._hybrid_weight, self.weight_forward_quantizer, self.weight_backward_quantizer,
                                                self._hybrid_input, self.input_forward_quantizer, self.input_backward_quantizer,
                                                self._hybrid_grad, self.grad_input_quantizer, self.grad_weight_quantizer)
        else:
            if self.training:
                print("Warning: quantization is disabled!!!")
            return super(QTConv2d, self).forward(input)
