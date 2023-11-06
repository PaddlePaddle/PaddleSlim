import paddle
import paddle.nn as nn
from paddleslim.lc.quantizers import NF4Quantizer
from .linear import Linear4bit


class NF4Linear(Linear4bit):
    def __init__(
            self,
            linear: nn.Linear,
            block_size=64,
            use_double_quant=False, ):
        super(NF4Linear, self).__init__(linear, quant_type="nf4")
        self.block_size = block_size
        self.double_quant = use_double_quant
        self.quantizer = NF4Quantizer(block_size, self.double_quant)

    def quantize(self, weight):
        quantized_weight = self.quantizer.quantize(weight)
        return {
            self.quant_weight_name: quantized_weight,
            self.quant_scale_name: self.quantizer.quant_scale,
            self.double_quant_scale_name: self.quantizer.double_quant_scale
        }

    def forward(self, x):
        self.quantizer.quant_scale = self.state_dict[self.quant_scale_name]
        self.quantizer.double_quant_scale = self.state_dict[
            self.double_quant_scale_name]
        return self.quantizer.matmul(x, self.quant_weight)
