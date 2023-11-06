import paddle
import paddle.nn as nn


class WeightQuantizationLinear(nn.Layer):
    def __init__(
            self,
            linear: paddle.nn.Linear, ):
        super().__init__()
        self.in_features = linear.weight.shape[0]
        self.out_features = linear.weight.shape[1]
        self.dtype = linear.dtype
        self.weight_name = linear.weight.name
        self.quant_weight_name = ".".join([self.weight_name, "quant_weight"])

    def forward(self, x):
        raise NotImplementedError()

    def quantize(self, weight) -> paddle.Tensor:
        raise NotImplementedError()
