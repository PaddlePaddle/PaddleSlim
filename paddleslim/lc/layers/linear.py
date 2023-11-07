# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle
import paddle.nn as nn


class WeightQuantizationLinear(nn.Layer):
    def __init__(
            self,
            linear: paddle.nn.Linear, ):
        super().__init__()
        self.linear = linear
        self.in_features = linear.weight.shape[0]
        self.out_features = linear.weight.shape[1]
        self.weight_name = linear.weight.name
        self.quant_weight_name = ".".join([self.weight_name, "quant_weight"])

    def forward(self, x):
        raise NotImplementedError()

    def quantize(self, weight) -> paddle.Tensor:
        raise NotImplementedError()


class Linear4bit(WeightQuantizationLinear):
    def __init__(
            self,
            linear: paddle.nn.Linear, quant_type="nf4"):
        super(Linear4bit, self).__init__(linear)
        self.quant_dtype = "int4"
        self.weight_dtype = "uint8"
        self.quant_scale_suffix = "_quant_scale"
        self.double_quant_scale_suffix = "_double_quant_scale"
    
        self.quant_weight = self.create_parameter(
            shape=[self.out_features // 2, self.in_features],
            attr=paddle.nn.initializer.Constant(value=0),
            dtype=self.weight_dtype,
            is_bias=False, )

    def forward(self, x):
        raise NotImplementedError()

    def quantize(self, weight) -> paddle.Tensor:
        raise NotImplementedError()
