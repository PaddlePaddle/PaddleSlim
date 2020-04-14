# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved. 
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
"dygraph transformer layers"

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Embedding, LayerNorm, Linear, Layer, Conv2D, BatchNorm


class ConvBN(fluid.dygraph.Layer):
    def __init__(self,
                 out_ch,
                 in_ch,
                 filter_size=3,
                 dilation=1,
                 act="relu",
                 is_test=False,
                 use_cudnn=True):
        super(ConvBN, self).__init__()
        conv_std = (2.0 / (filter_size**2 * in_ch))**0.5
        conv_param = fluid.ParamAttr(
            initializer=fluid.initializer.Normal(0.0, conv_std))

        self.conv_layer = Conv2D(
            in_ch,
            out_ch, [filter_size, 1],
            dilation=dilation,
            padding=[(filter_size - 1) // 2, 0],
            param_attr=conv_param,
            bias_attr=False,
            act=None,
            use_cudnn=use_cudnn)
        self.bn_layer = BatchNorm(out_ch, act=act, is_test=is_test)

    def forward(self, inputs):
        conv = self.conv_layer(inputs)
        bn = self.bn_layer(conv)
        return bn


class EncoderSubLayer(Layer):
    """
    EncoderSubLayer
    """

    def __init__(self, name=""):

        super(EncoderSubLayer, self).__init__()
        self.name = name
        self.conv0 = ConvBN(1, 1, filter_size=5)
        self.conv1 = ConvBN(1, 1, filter_size=5)
        self.conv2 = ConvBN(1, 1, filter_size=5)

    def forward(self, enc_input):
        """
        forward
        :param enc_input:
        :param attn_bias:
        :return:
        """
        tmp = self.conv0(enc_input)
        tmp = self.conv1(tmp)
        tmp = self.conv2(tmp)
        return tmp


class EncoderLayer(Layer):
    """
    encoder
    """

    def __init__(self, n_layer, d_model=128, name=""):

        super(EncoderLayer, self).__init__()
        self._encoder_sublayers = list()
        self._n_layer = n_layer
        self._d_model = d_model

        for i in range(n_layer):
            self._encoder_sublayers.append(
                self.add_sublayer(
                    'esl_%d' % i,
                    EncoderSubLayer(name=name + '_layer_' + str(i))))

    def forward(self, enc_input):
        """
        forward
        :param enc_input:
        :param attn_bias:
        :return:
        """
        tmp = fluid.layers.reshape(enc_input,
                                   [-1, 1, enc_input.shape[1], self._d_model])
        outputs = []
        for i in range(self._n_layer):
            tmp = self._encoder_sublayers[i](tmp)
            enc_output = fluid.layers.reshape(
                tmp, [-1, enc_input.shape[1], self._d_model])
            outputs.append(enc_output)
        return outputs
