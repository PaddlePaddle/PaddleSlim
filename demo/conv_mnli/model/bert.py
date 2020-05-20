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

import six
import json
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Embedding, LayerNorm, Linear, to_variable, Layer, guard
from .transformer_encoder import ConvBNEncoderLayer
from .mobilenet_v1 import MobileNetV1


class BertModelLayer(Layer):
    def __init__(self,
                 emb_size=128,
                 hidden_size=768,
                 voc_size=30522,
                 max_position_seq_len=512,
                 sent_types=2,
                 return_pooled_out=True,
                 initializer_range=1.0,
                 conv_type="mobilenet",
                 use_fp16=False):
        super(BertModelLayer, self).__init__()

        self._emb_size = emb_size
        self._hidden_size = hidden_size
        self._voc_size = voc_size
        self._max_position_seq_len = max_position_seq_len
        self._sent_types = sent_types
        self.return_pooled_out = return_pooled_out

        self._word_emb_name = "s_word_embedding"
        self._pos_emb_name = "s_pos_embedding"
        self._sent_emb_name = "s_sent_embedding"
        self._dtype = "float16" if use_fp16 else "float32"

        self._conv_type = conv_type
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=initializer_range)

        self._src_emb = Embedding(
            size=[self._voc_size, self._emb_size],
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name, initializer=self._param_initializer),
            dtype=self._dtype)

        self._pos_emb = Embedding(
            size=[self._max_position_seq_len, self._emb_size],
            param_attr=fluid.ParamAttr(
                name=self._pos_emb_name, initializer=self._param_initializer),
            dtype=self._dtype)

        self._sent_emb = Embedding(
            size=[self._sent_types, self._emb_size],
            param_attr=fluid.ParamAttr(
                name=self._sent_emb_name, initializer=self._param_initializer),
            dtype=self._dtype)

        self._emb_fac = Linear(
            input_dim=self._emb_size,
            output_dim=self._hidden_size,
            param_attr=fluid.ParamAttr(name="s_emb_factorization"))

        #        self.pooled_fc = Linear(
        #            input_dim=self._hidden_size,
        #            output_dim=self._hidden_size,
        #            param_attr=fluid.ParamAttr(
        #                name="s_pooled_fc.w_0", initializer=self._param_initializer),
        #            bias_attr="s_pooled_fc.b_0",
        #            act="tanh")
        if self._conv_type == "conv_bn":
            self._encoder = ConvBNEncoderLayer(
                hidden_size=self._hidden_size, name="encoder")
        elif self._conv_type == "mobilenet":
            self._encoder = MobileNetV1(1, 3)

    def forward(self, src_ids, position_ids, sentence_ids):
        src_emb = self._src_emb(src_ids)
        pos_emb = self._pos_emb(position_ids)
        sent_emb = self._sent_emb(sentence_ids)

        emb_out = src_emb + pos_emb
        emb_out = emb_out + sent_emb

        emb_out = self._emb_fac(emb_out)
        emb_out = fluid.layers.reshape(
            emb_out, shape=[-1, 1, emb_out.shape[1], emb_out.shape[2]])
        enc_output = self._encoder(emb_out)

        #        if not self.return_pooled_out:
        #            return enc_output
        #        next_sent_feat = fluid.layers.slice(
        #                input=enc_output, axes=[2], starts=[0], ends=[1])
        #        next_sent_feat = fluid.layers.reshape(
        #                next_sent_feat, shape=[-1, self._hidden_size])
        #        next_sent_feat = self.pooled_fc(next_sent_feat)

        return enc_output
