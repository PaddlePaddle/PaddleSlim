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
                 fix_emb=False,
                 t=5.0):
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
        self.T = t
        print(
            "----------------------load teacher model and test----------------------------------------"
        )
        #        self.teacher = BERTClassifier(
        #            num_labels, model_path=self._teacher_model)
        #        self.teacher.test(self._data_dir)
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

        #        for s_emb, t_emb in zip(self.student.emb_names(),
        #                                self.teacher.emb_names()):
        #            t_emb.stop_gradient = True
        #            if fix_emb:
        #                s_emb.stop_gradient = True
        #            print(
        #                "Assigning embedding[{}] from teacher to embedding[{}] in student.".
        #                format(t_emb.name, s_emb.name))
        #            fluid.layers.assign(input=t_emb, output=s_emb)
        #            print(
        #                "Assigned embedding[{}] from teacher to embedding[{}] in student.".
        #                format(t_emb.name, s_emb.name))

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
        self.labels = None

    def test(self, a, b):
        return self.student(a, b)

    def arch_parameters(self):
        return self.student.arch_parameters()

    def genotype(self):
        return self.arch_parameters()

    def ce(self, logits):
        logits = np.exp(logits - np.max(logits))
        logits = logits / logits.sum(axis=0)
        return logits

    def forward(self, a_ids, b_ids, labels, t_logits, t_losses):
        labels = fluid.dygraph.to_variable(labels)
        t_logits = fluid.layers.transpose(
            t_logits, [1, 0, 2])  # [layers, batch, logit_size]
        t_losses = fluid.layers.reduce_mean(t_losses, dim=0)
        t_losses = fluid.layers.reshape(t_losses, [-1, 1])
        s_logits = self.student(a_ids, b_ids, flops=[], model_size=[])

        #        self.teacher.eval()
        #        total_loss, t_logits, t_losses, accuracys, num_seqs = self.teacher(
        #            data_ids)
        #        self.teacher.train()

        # define kd loss

        t_layers_n = t_logits.shape[0]
        s_layers_n = len(s_logits)
        kd_losses = []
        #
        t_indexes = []
        for i in range(s_layers_n):
            j = int(np.ceil(i * (float(t_layers_n) / s_layers_n)))
            t_indexes.append(j)
        t_indexes = fluid.dygraph.to_variable(np.array(t_indexes))

        t_losses = fluid.layers.gather(t_losses, t_indexes)
        t_losses = fluid.layers.softmax(t_losses, axis=0)

        t_logits = fluid.layers.gather(t_logits, t_indexes)

        t_probs = fluid.layers.softmax(t_logits / self.T)

        s_logits = fluid.layers.stack(
            s_logits, axis=0)  # [layers_n, batch, logit_size]
        s_probs = fluid.layers.softmax(s_logits)

        kd_loss = fluid.layers.cross_entropy(
            input=s_probs, label=t_probs,
            soft_label=True)  # [layers_n, batch, 1]

        kd_loss = fluid.layers.reduce_mean(kd_loss, dim=1)  # [layers_n, 1]
        kd_loss = fluid.layers.reduce_sum(kd_loss * t_losses)

        ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
            logits=s_logits[-1], label=labels, return_softmax=True)

        ce_loss = fluid.layers.mean(x=ce_loss)
        num_seqs = fluid.layers.create_tensor(dtype='int64')
        accuracy = fluid.layers.accuracy(
            input=probs, label=labels, total=num_seqs)

        loss = (1 - self._gamma) * ce_loss + self._gamma * kd_loss
        #        return ce_loss, accuracy, ce_loss, ce_loss
        return loss, accuracy, ce_loss, kd_loss
