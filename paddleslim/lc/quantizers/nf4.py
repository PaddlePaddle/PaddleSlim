import paddle
from .base_quantizer import BaseQuantizer
import paddleslim_ops


class NF4Quantizer(BaseQuantizer):
    dtype = "int4"

    def __init__(self, block_size=64, double_quant=False):
        super(BaseQuantizer, self).__init__()
        self.block_size = block_size
        self.double_quant = double_quant
        self.quant_scale = None
        self.double_quant_scale = None

    def quantize(self, x: paddle.Tensor):
        out, abs_max = paddleslim_ops.quantize_nf4(
            x, block_size=self.block_size)
        self.quant_scale = abs_max
        return out

    def dequantize(self, x: paddle.Tensor, dtype: int):
        return paddleslim_ops.dequantize_nf4(
            x, self.quant_scale, block_size=self.block_size, dtype=dtype)

    def matmul(self, x: paddle.Tensor, y: paddle.Tensor, bias: paddle.Tensor):
        return x @ self.dequantize(y) + bias
