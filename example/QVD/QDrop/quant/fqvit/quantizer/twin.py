import paddle
from .base import BaseQuantizer


class TwinQuantizer(BaseQuantizer):

    def __init__(self, bit_type, observer, module_type):
        super(TwinQuantizer, self).__init__(bit_type, observer, module_type)
        self.a_neg_interval = None
        self.a_interval = None
        self.bit_type = bit_type

    def update_quantization_params(self, *args, **kwargs):
        self.a_interval, self.a_neg_interval = (self.observer.
            get_quantization_params(*args, **kwargs))

    def quant(self, inputs, scale=None, zero_point=None):
        x = inputs
        int_input_pos = (x / self.a_interval).round_().clip_(min=0, max=
            self.bit_type.upper_bound - 1)
        int_input_pos = int_input_pos.detach().to('uint8') + 128
        int_input_neg = (x / self.a_neg_interval).round_().clip_(min=-self.
            bit_type.lower_bound + 1, max=0).abs()
        int_input_neg = int_input_neg.detach().to('uint8')
        int_input = (int_input_pos + int_input_neg).cuda(blocking=True)
        return int_input
        """
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = inputs / scale + zero_point
        outputs = outputs.round().clamp(
            self.bit_type.lower_bound, self.bit_type.upper_bound)
        return outputs
        """

    def dequantize(self, inputs, a_interval=None, zero_point=None):
        if a_interval is None:
            a_interval = self.a_interval
        a_neg_interval = self.a_neg_interval
        range_shape = self.get_reshape_range(inputs)
        input_pos = (inputs - 128) * a_interval
        input_neg = inputs * a_neg_interval
        out = (input_pos + input_neg).cuda(blocking=True)
        return out
