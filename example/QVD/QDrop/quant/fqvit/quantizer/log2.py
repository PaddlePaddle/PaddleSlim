import paddle
from .base import BaseQuantizer


class Log2Quantizer(BaseQuantizer):

    def __init__(self, bit_type, observer, module_type):
        super(Log2Quantizer, self).__init__(bit_type, observer, module_type)
        self.softmax_mask = None

    def quant(self, inputs):
        rounds = paddle.round(-1 * inputs.log2())
        self.softmax_mask = rounds >= 2 ** self.bit_type.bits
        outputs = paddle.clip(x=rounds, min=0, max=2 ** self.bit_type.bits - 1)
        return outputs

    def dequantize(self, inputs):
        outputs = 2 ** (-1 * inputs)
        outputs[self.softmax_mask] = 0
        return outputs
