#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""BERT fine-tuning in Paddle Dygraph Mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import sys
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf8')
import ast
import time
import argparse
import numpy as np
import multiprocessing
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable, Layer, Linear
from .reader.cls import *
from .model.bert import BertModelLayer
from .optimization import Optimizer
from .utils.init import init_from_static_model
from paddleslim.teachers.bert import BERTClassifier

__all__ = ["AdaBERTClassifier"]


class AdaBERTClassifier(Layer):
    def __init__(self, num_labels, n_layer=8, emb_size=768,
                 teacher_model=None):
        super(AdaBERTClassifier, self).__init__()
        self._n_layer = n_layer
        self._num_labels = num_labels
        self._emb_size = emb_size
        print(
            "----------------------load teacher model and test----------------------------------------"
        )
        self.teacher = BERTClassifier(num_labels, model_path=teacher_model)
        #        self.teacher.test("/work/PaddleSlim/demo/bert/data/glue_data/MNLI/")
        print(
            "----------------------finish load teacher model and test----------------------------------------"
        )
        self.student = BertModelLayer(
            n_layer=self._n_layer, emb_size=self._emb_size)

        self.cls_fc = list()
        for i in range(self._n_layer):
            fc = Linear(
                input_dim=self._emb_size,
                output_dim=self._num_labels,
                param_attr=fluid.ParamAttr(
                    name="s_cls_out_%d_w" % i,
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
                bias_attr=fluid.ParamAttr(
                    name="s_cls_out_%d_b" % i,
                    initializer=fluid.initializer.Constant(0.)))
            fc = self.add_sublayer("cls_fc_%d" % i, fc)
            self.cls_fc.append(fc)

    def forward(self, data_ids):
        src_ids = data_ids[0]
        position_ids = data_ids[1]
        sentence_ids = data_ids[2]
        return self.student(src_ids, position_ids, sentence_ids)

    def arch_parameters(self):
        return self.student.arch_parameters()

    def genotype(self):
        return self.arch_parameters()

    def loss(self, data_ids, beta=4, gamma=0.8):
        T = 1.0
        src_ids = data_ids[0]
        position_ids = data_ids[1]
        sentence_ids = data_ids[2]
        input_mask = data_ids[3]
        labels = data_ids[4]
        flops = []
        model_size = []
        enc_outputs, next_sent_feats, k_i = self.student(
            src_ids,
            position_ids,
            sentence_ids,
            flops=flops,
            model_size=model_size)

        self.teacher.eval()
        total_loss, t_logits, t_losses, accuracys, num_seqs = self.teacher(
            data_ids)

        # define kd loss
        kd_losses = []

        kd_weights = []
        for i in range(len(next_sent_feats)):
            j = int(np.ceil(i * (float(len(t_logits)) / len(next_sent_feats))))
            kd_weights.append(t_losses[j].numpy())
        kd_weights = 1 / np.array(kd_weights)

        kd_weights = np.exp(kd_weights - np.max(kd_weights))

        kd_weights = kd_weights / kd_weights.sum(axis=0)

        for i in range(len(next_sent_feats)):
            j = int(np.ceil(i * (float(len(t_logits)) / len(next_sent_feats))))
            t_logit = t_logits[j]
            s_sent_feat = next_sent_feats[i]
            fc = self.cls_fc[i]
            s_sent_feat = fluid.layers.dropout(
                x=s_sent_feat,
                dropout_prob=0.1,
                dropout_implementation="upscale_in_train")
            s_logits = fc(s_sent_feat)

            t_probs = fluid.layers.softmax(t_logit)
            s_probs = fluid.layers.softmax(s_logits)
            t_probs.stop_gradient = False
            kd_loss = t_probs * fluid.layers.log(s_probs / T)
            kd_loss = fluid.layers.reduce_sum(kd_loss, dim=1)
            kd_loss = kd_loss * kd_weights[i]
            kd_losses.append(kd_loss)

        kd_loss = fluid.layers.sum(kd_losses)
        kd_loss = fluid.layers.reduce_mean(kd_loss, dim=0)

        # define ce loss
        ce_loss = fluid.layers.cross_entropy(s_probs, labels)
        ce_loss = fluid.layers.reduce_mean(ce_loss) * k_i

        # define e loss
        model_size = fluid.layers.sum(
            model_size) / self.student.max_model_size()
        flops = fluid.layers.sum(flops) / self.student.max_flops()
        e_loss = (len(next_sent_feats) * k_i / self._n_layer) * (
            flops + model_size)
        # define total loss
        loss = (1 - gamma) * ce_loss - gamma * kd_loss + beta * e_loss
        print("ce_loss: {}; kd_loss: {}; e_loss: {}".format((
            1 - gamma) * ce_loss.numpy(), -gamma * kd_loss.numpy(), beta *
                                                            e_loss.numpy()))
        return loss
