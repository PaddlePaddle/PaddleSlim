import sys
sys.path.append("../../")
import numpy as np
import unittest
import paddle
from paddleslim.lc.layers import NF4Linear, FP4Linear
from paddleslim.lc.quantizers.quant_func import quantize_nf4, quantize_fp4, dequantize_nf4, dequantize_fp4, quantize_8bit, dequantize_8bit

class NF4(unittest.TestCase):
    def setUp(self):
        self.quant_type = "nf4"
        self.blocksize = 64

    def test_nf4_fp16(self):
        a = paddle.uniform([2, 64], dtype="float16")
        nf4_a, scale_a = quantize_nf4(a, self.blocksize)
        fp16_a = dequantize_nf4(nf4_a, scale_a, self.blocksize).cast("float16")

class FP4(unittest.TestCase):
    def setUp(self):
        self.quant_type = "fp4"
        self.blocksize = 64

    def test_fp4_fp16(self):
        a = paddle.uniform([2, 64], dtype="float16")
        nf4_a, scale_a = quantize_fp4(a, self.blocksize)
        fp16_a = dequantize_fp4(nf4_a, scale_a, self.blocksize).cast("float16")

class BIT8(unittest.TestCase):
    def setUp(self):
        self.quant_type = "fp8"
        self.blocksize = 64

    def test_fp8_fp16(self):
        a = paddle.uniform([2, 64], dtype="float16")
        nf4_a, scale_a = quantize_8bit(a, None, self.blocksize, quant_type="fp8")
        fp16_a = dequantize_8bit(nf4_a, None, scale_a, self.blocksize, quant_type="fp8").cast("float16")

    def test_dynamic_fp8_fp16(self):
        a = paddle.uniform([2, 64], dtype="float16")
        nf4_a, scale_a = quantize_8bit(a, None, self.blocksize, quant_type="dynamic_fp8")
        fp16_a = dequantize_8bit(nf4_a, None, scale_a, self.blocksize, quant_type="dynamic_fp8").cast("float16")

if __name__ == '__main__':
    unittest.main()


