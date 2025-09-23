import paddle
import paddle.nn as nn

from .base import BaseQuantizer


class UniformQuantizer(BaseQuantizer):
    def __init__(self, bit_type, observer, module_type):
        super(UniformQuantizer, self).__init__(bit_type, observer, module_type)
        self.scale = None
        self.zero_point = None

    def update_quantization_params(self, *args, **kwargs):
        # 获取量化参数
        self.scale, self.zero_point = self.observer.get_quantization_params(*args, **kwargs)

    def quant(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = paddle.reshape(scale, range_shape)
        zero_point = paddle.reshape(zero_point, range_shape)
        
        outputs = inputs / scale + zero_point
        outputs = paddle.round(outputs)
        outputs = paddle.clip(outputs, min=self.bit_type.lower_bound, max=self.bit_type.upper_bound)
        
        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = paddle.reshape(scale, range_shape)
        zero_point = paddle.reshape(zero_point, range_shape)
        outputs = (inputs - zero_point) * scale
        return outputs

