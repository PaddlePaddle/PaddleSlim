import paddle
import paddle.nn as nn
from paddleslim.lc.quantizers import FP4Quantizer
from .linear import Linear4bit


class FP4Linear(Linear4bit):
    def __init__(
            self,
            linear: nn.Linear,
            block_size=64,
            use_double_quant=False, ):
        super(FP4Linear, self).__init__(linear, quant_type="fp4")
        self.block_size = block_size
        self.double_quant = use_double_quant
        self.quantizer = FP4Quantizer(block_size, self.double_quant)

    def quantize(self):
        quantized_weight = self.quantizer.quantize(self.linear.weight).reshape([self.out_features // 2, self.in_features])
        self.quant_weight.set_value(quantized_weight)
        return {
            self.quant_weight_name: quantized_weight,
            self.quantizer.quant_state: self.quantizer.state,
        }

    def forward(self, x):
        return self.quantizer.matmul(x, self.quant_weight, self.linear.bias)
