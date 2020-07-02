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
from paddleslim.teachers.bert import BERTClassifier

__all__ = ["AdaBERTClassifier"]


class AdaBERTClassifier(Layer):
    def __init__(self,
                 num_labels,
                 gamma=0.8,
                 beta=4,
                 teacher_model=None,
                 data_dir=None,
                 fix_emb=False,
                 student_bert_config=None,
                 t=5.0):
        super(AdaBERTClassifier, self).__init__()
        self._gamma = gamma
        self._beta = beta
        self._teacher_model = teacher_model
        self._data_dir = data_dir
        self.T = t
        print(
            "----------------------load teacher model and test----------------------------------------"
        )
        self.teacher = BERTClassifier(
            num_labels,
            model_path=self._teacher_model,
            return_pooled_out=False)
        #        self.teacher.test(self._data_dir)
        print(
            "----------------------finish load teacher model and test----------------------------------------"
        )

        self.student = BERTClassifier(
            num_labels,
            return_pooled_out=False,
            bert_config=student_bert_config,
            name="student")

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

    def val(self, data_ids):
        src_ids = data_ids[0]
        position_ids = data_ids[1]
        sentence_ids = data_ids[2]
        return self.student(data_ids)

    def forward(self, data_ids):
        src_ids = data_ids[0]
        position_ids = data_ids[1]
        sentence_ids = data_ids[2]
        input_mask = data_ids[3]
        labels = data_ids[4]

        if self._gamma != 0:
            self.teacher.eval()
            t_enc_outputs, t_logits, _, _ = self.teacher(data_ids)

        self.student.train()
        s_enc_outputs, s_logits, _, _ = self.student(data_ids)

        ## ce loss of student
        ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
            logits=s_logits, label=labels, return_softmax=True)
        ce_loss = fluid.layers.mean(x=ce_loss)
        num_seqs = fluid.layers.create_tensor(dtype='int64')
        accuracy = fluid.layers.accuracy(
            input=probs, label=labels, total=num_seqs)

        if self._gamma != 0:
            # define kd loss
            ## The prediction layers distillation loss
            t_logits = fluid.layers.softmax(t_logits / self.T)
            t_logits.stop_gradient = True
            kd_loss = fluid.layers.softmax_with_cross_entropy(
                s_logits, t_logits, soft_label=True)
            kd_loss = fluid.layers.mean(x=kd_loss)

            ## The internal layers distillation loss
            for s_v, t_v in zip(s_enc_outputs, t_enc_outputs):
                t_v.stop_gradient = True
                mse = fluid.layers.mse_loss(s_v, t_v)
                kd_loss = kd_loss + mse

            loss = (1 - self._gamma) * ce_loss + self._gamma * kd_loss
        else:
            loss = ce_loss
            kd_loss = ce_loss
        #        return ce_loss, accuracy, None, None
        return loss, accuracy, ce_loss, kd_loss
