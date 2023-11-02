import paddle
import paddle.nn as nn
from paddleslim.lc.quantizers import NF4Quantizer
from .linear import WeightQuantizationLinear


class NF4Linear(WeightQuantizationLinear):
    quant_dtype = "int4"
    weight_dtype = "int8"
    quant_scale_suffix = "quant_scale"
    double_quant_scale_suffix = "double_quant_scale"

    def __init__(
            self,
            linear: nn.Linear,
            block_size=64,
            use_double_quant=False, ):
        super(NF4Linear, self).__init__(linear)
        self.block_size = block_size
        self.double_quant = use_double_quant
        self.quantizer = NF4Quantizer(block_size, use_double_quant)
        # PaddlePaddle dosen't support Int4 data type, one Int8 data represents two Int4 data.
        self.quant_weight = self.create_parameter(
            shape=[self.out_features // 2, self.in_features],
            attr=paddle.ParamAttr(self.quant_weight_name),
            dtype=NF4Linear.weight_dtype,
            is_bias=False, )

        self.quant_scale_name = ".".join(
            [self.weight_name, NF4Linear.quant_scale_suffix])
        self.quant_scale = self.create_parameter(
            shape=[self.out_features],
            attr=paddle.ParamAttr(self.quant_scale_name),
            dtype="float32",  # to be fixed
            is_bias=False, )
        if self.double_quant:
            self.double_quant_scale_name = ".".join(
                [self.weight_name, NF4Linear.double_quant_scale_suffix])
            self.double_quant_scale = self.create_parameter(
                shape=[self.out_features],
                attr=paddle.ParamAttr(self.double_quant_scale_name),
                dtype="float32",
                is_bias=False, )

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
