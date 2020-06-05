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
    def __init__(self,
                 num_labels,
                 n_layer=8,
                 emb_size=768,
                 hidden_size=768,
                 gamma=0.8,
                 beta=4,
                 conv_type="conv_bn",
                 search_layer=False,
                 teacher_model=None,
                 data_dir=None,
                 use_fixed_gumbel=False,
                 gumbel_alphas=None,
                 fix_emb=False):
        super(AdaBERTClassifier, self).__init__()
        self._n_layer = n_layer
        self._num_labels = num_labels
        self._emb_size = emb_size
        self._hidden_size = hidden_size
        self._gamma = gamma
        self._beta = beta
        self._conv_type = conv_type
        self._search_layer = search_layer
        self._teacher_model = teacher_model
        self._data_dir = data_dir
        self.use_fixed_gumbel = use_fixed_gumbel
        print(
            "----------------------load teacher model and test----------------------------------------"
        )
        self.teacher = BERTClassifier(
            num_labels, model_path=self._teacher_model)
        self.teacher.test(self._data_dir)
        print(
            "----------------------finish load teacher model and test----------------------------------------"
        )
        self.student = BertModelLayer(
            n_layer=self._n_layer,
            emb_size=self._emb_size,
            hidden_size=self._hidden_size,
            conv_type=self._conv_type,
            search_layer=self._search_layer,
            use_fixed_gumbel=self.use_fixed_gumbel,
            gumbel_alphas=gumbel_alphas)

        for s_emb, t_emb in zip(self.student.emb_names(),
                                self.teacher.emb_names()):
            t_emb.stop_gradient = True
            if fix_emb:
                s_emb.stop_gradient = True
            print(
                "Assigning embedding[{}] from teacher to embedding[{}] in student.".
                format(t_emb.name, s_emb.name))
            fluid.layers.assign(input=t_emb, output=s_emb)
            print(
                "Assigned embedding[{}] from teacher to embedding[{}] in student.".
                format(t_emb.name, s_emb.name))

        self.cls_fc = list()
        for i in range(self._n_layer):
            fc = Linear(
                input_dim=self._hidden_size,
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

    def loss(self, data_ids):
        src_ids = data_ids[0]
        position_ids = data_ids[1]
        sentence_ids = data_ids[2]
        input_mask = data_ids[3]
        labels = data_ids[4]

        s_logits = self.student(
            src_ids, position_ids, sentence_ids, flops=[], model_size=[])

        self.teacher.eval()
        total_loss, t_logits, t_losses, accuracys, num_seqs = self.teacher(
            data_ids)
        self.teacher.train()

        # define kd loss
        kd_losses = []

        kd_weights = []
        for i in range(len(s_logits)):
            j = int(np.ceil(i * (float(len(t_logits)) / len(s_logits))))
            kd_weights.append(t_losses[j].numpy())
        kd_weights = 1 / np.array(kd_weights)

        kd_weights = np.exp(kd_weights - np.max(kd_weights))

        kd_weights = kd_weights / kd_weights.sum(axis=0)
        s_probs = None
        for i in range(len(s_logits)):
            j = int(np.ceil(i * (float(len(t_logits)) / len(s_logits))))
            t_logit = t_logits[j]
            s_logit = s_logits[i]
            t_logit.stop_gradient = True
            t_probs = fluid.layers.softmax(t_logit)
            s_probs = fluid.layers.softmax(s_logit)
            kd_loss = t_probs * fluid.layers.log(s_probs / 1.0)
            kd_loss = fluid.layers.reduce_sum(kd_loss, dim=1)
            kd_loss = fluid.layers.mean(kd_loss)
            #            print("kd_loss[{}] = {}; kd_weights[{}] = {}".format(i, kd_loss.numpy(), i, kd_weights[i]))
            #            tmp = kd_loss * kd_weights[i]
            tmp = fluid.layers.scale(kd_loss, scale=kd_weights[i])
            #            print("kd_loss[{}] = {}".format(i, tmp.numpy()))
            kd_losses.append(tmp)

        kd_loss = fluid.layers.sum(kd_losses)

        #        print("kd_loss = {}".format(kd_loss.numpy()))

        ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
            logits=s_logits[-1], label=labels, return_softmax=True)
        ce_loss = fluid.layers.mean(x=ce_loss)
        num_seqs = fluid.layers.create_tensor(dtype='int64')
        accuracy = fluid.layers.accuracy(
            input=probs, label=labels, total=num_seqs)

        loss = (1 - self._gamma) * ce_loss - self._gamma * kd_loss
        #        return ce_loss, accuracy, None, None
        return loss, accuracy, ce_loss, kd_loss
