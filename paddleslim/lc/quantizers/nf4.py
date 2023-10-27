import paddle
from .base_quantizer import BaseQuantizer


class NF4Quantizer(BaseQuantizer):
    dtype = "int4"

    def __init__(self, block_size=64, double_quant=False):
        super(BaseQuantizer, self).__init__()
        self.block_size = block_size
        self.double_quant = double_quant
        self.quant_scale = None
        self.double_quant_scale = None

    def quantize(self, x: paddle.Tensor):
        return x

    def dequantize(self, x: paddle.Tensor):
        return x

    def matmul(self, x: paddle.Tensor, y: paddle.Tensor, bias: paddle.Tensor):
        return x @ self.dequantize(y) + bias
