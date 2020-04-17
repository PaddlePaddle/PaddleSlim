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
from .transformer_encoder import EncoderLayer


class BertModelLayer(Layer):
    def __init__(self,
                 emb_size=768,
                 n_layer=12,
                 voc_size=30522,
                 max_position_seq_len=512,
                 sent_types=2,
                 return_pooled_out=True,
                 initializer_range=1.0,
                 use_fp16=False):
        super(BertModelLayer, self).__init__()

        self._emb_size = emb_size
        self._n_layer = n_layer
        self._voc_size = voc_size
        self._max_position_seq_len = max_position_seq_len
        self._sent_types = sent_types
        self.return_pooled_out = return_pooled_out

        self._word_emb_name = "s_word_embedding"
        self._pos_emb_name = "s_pos_embedding"
        self._sent_emb_name = "s_sent_embedding"
        self._dtype = "float16" if use_fp16 else "float32"

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

        self.pooled_fc = Linear(
            input_dim=self._emb_size,
            output_dim=self._emb_size,
            param_attr=fluid.ParamAttr(
                name="s_pooled_fc.w_0", initializer=self._param_initializer),
            bias_attr="s_pooled_fc.b_0",
            act="tanh")

        self._encoder = EncoderLayer(
            n_layer=self._n_layer, d_model=self._emb_size)

    def max_flops(self):
        return self._encoder.max_flops

    def max_model_size(self):
        return self._encoder.max_model_size

    def arch_parameters(self):
        return [self._encoder.alphas]

    def forward(self,
                src_ids,
                position_ids,
                sentence_ids,
                flops=[],
                model_size=[]):
        """
        forward
        """
        src_emb = self._src_emb(src_ids)
        pos_emb = self._pos_emb(position_ids)
        sent_emb = self._sent_emb(sentence_ids)

        emb_out = src_emb + pos_emb
        emb_out = emb_out + sent_emb

        enc_outputs, k_i = self._encoder(
            emb_out, flops=flops, model_size=model_size)

        if not self.return_pooled_out:
            return enc_outputs
        next_sent_feats = []
        for enc_output in enc_outputs:
            next_sent_feat = fluid.layers.slice(
                input=enc_output, axes=[1], starts=[0], ends=[1])
            next_sent_feat = self.pooled_fc(next_sent_feat)
            next_sent_feat = fluid.layers.reshape(
                next_sent_feat, shape=[-1, self._emb_size])
            next_sent_feats.append(next_sent_feat)

        return enc_outputs, next_sent_feats, k_i
