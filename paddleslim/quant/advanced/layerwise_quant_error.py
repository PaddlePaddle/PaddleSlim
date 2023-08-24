# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import numpy as np
from .utils import compute_scales
from .metrics import mse_loss

__all__ = ['LayerWiseQuantError']


class LayerWiseQuantError(nn.Layer):
    def __init__(self,
                 layer,
                 weight_bits=8,
                 act_bits=8,
                 weight_quant_method='abs_max_channel_wise',
                 act_quant_method='abs_max',
                 loss_function=mse_loss):
        '''
        LayerWiseQuantError computes the loss bewteen the output of the layer and the outout of the quantized layer.
        
        Args:
        layer (paddle.nn.Layer): Layer object.
        quant_bits (int, optional): Number of bits to quantize the weight. Default: 8.
        act_bits (int, optional): Number of bits to quantize the activation. Default: 8.
        weight_quant_method (str, optional): The method of weight quantization. Choosen from abs_max, abs_max_channel_wise and avg. Default: abs_max_channel_wise.
        act_quant_method (str, optional): The method of activation quantization. Choosen from abs_max, avg. Default: abs_max.

        Examples:
        .. code-block:: python
        
        from paddleslim.quant.advanced import GPTQ
        for cur_name, cur_layer in model.named_sublayers():
            if type(cur_layer) == paddle.nn.Linear:
                gptq_layer = LayerWiseQuantError(cur_layer)

        for data in dataloader():
            model(data)

        for cur_name, cur_layer in model.named_sublayers():
            if type(cur_layer) == LayerWiseQuantError:
                print(cur_name, cur_layer.losses.mean())
        '''
        super(LayerWiseQuantError, self).__init__()
        self.layer = layer
        self.weight = layer.weight
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.weight_method = weight_quant_method
        self.act_method = act_quant_method
        self.loss_function = loss_function
        self.losses = []
        self.loss = None

    def forward(self, input):
        act = input[0] if type(input) == tuple else input
        origin_out = paddle.matmul(act, self.weight)
        bnt = (1 << (self.weight_bits - 1)) - 1
        quant_scale = compute_scales(self.weight, method=self.weight_method)
        quant_weight = paddle.clip(
            paddle.round(self.weight / quant_scale * bnt), -bnt - 1, bnt)
        quant_dequant_weight = quant_weight / bnt * quant_scale

        bnt = (1 << (self.act_bits - 1)) - 1
        quant_scale = compute_scales(act, method=self.act_method)
        quant_act = paddle.clip(
            paddle.round(act / quant_scale * bnt), -bnt - 1, bnt)
        quant_dequant_act = quant_act / bnt * quant_scale
        quant_out = paddle.matmul(quant_dequant_act, quant_dequant_weight)
        loss = self.loss_function(origin_out, quant_out).cast('float32')
        self.losses.append(loss)
        self.loss = paddle.to_tensor(self.losses, dtype='float32').mean()
        return self.layer(input)
