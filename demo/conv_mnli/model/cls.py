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
from paddle.fluid.dygraph import Linear, Layer
from paddle.fluid.dygraph import to_variable

from .bert import BertModelLayer


class ClsModelLayer(Layer):
    """
    classify model
    """

    def __init__(self, num_labels, return_pooled_out=True):
        super(ClsModelLayer, self).__init__()
        self._hiden_size = 768

        self.bert_layer = BertModelLayer(
            emb_size=128,
            hidden_size=self._hiden_size,
            voc_size=30522,
            max_position_seq_len=512,
            sent_types=2,
            return_pooled_out=True,
            initializer_range=1.0,
            conv_type="mobilenet",
            use_fp16=False)

#        self.cls_fc = Linear(
#                input_dim=self._hiden_size,
#                output_dim=num_labels,
#                param_attr=fluid.ParamAttr(
#                    name="cls_out_w",
#                    initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
#                bias_attr=fluid.ParamAttr(
#                    name="cls_out_b",
#                    initializer=fluid.initializer.Constant(0.)))

    def forward(self, data_ids):
        """
        forward
        """
        src_ids = to_variable(data_ids[0])
        position_ids = to_variable(data_ids[1])
        sentence_ids = to_variable(data_ids[2])
        input_mask = data_ids[3]
        labels = to_variable(data_ids[4])

        logits = self.bert_layer(src_ids, position_ids, sentence_ids)
        #        cls_feat = fluid.layers.dropout(
        #                x=next_sent_feat,
        #                dropout_prob=0.1,
        #                dropout_implementation="upscale_in_train")
        #        logit = self.cls_fc(cls_feat)

        ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
            logits=logits, label=labels, return_softmax=True)
        loss = fluid.layers.mean(x=ce_loss)

        num_seqs = fluid.layers.create_tensor(dtype='int64')
        accuracy = fluid.layers.accuracy(
            input=probs, label=labels, total=num_seqs)

        return loss, accuracy, num_seqs
