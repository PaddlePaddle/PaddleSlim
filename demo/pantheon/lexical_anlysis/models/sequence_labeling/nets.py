#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
The function lex_net(args) define the lexical analysis network structure
"""
import sys
import os
import math

import paddle.fluid as fluid
from paddle.fluid.initializer import NormalInitializer


def lex_net(word, args, vocab_size, num_labels, teacher_crf_decode=None, for_infer=True,target=None):
    """
    define the lexical analysis network structure
    word: stores the input of the model
    for_infer: a boolean value, indicating if the model to be created is for training or predicting.

    return:
        for infer: return the prediction
        otherwise: return the prediction
    """
    word_emb_dim = args.word_emb_dim
    grnn_hidden_dim = args.grnn_hidden_dim
    emb_lr = args.emb_learning_rate if 'emb_learning_rate' in dir(args) else 1.0
    crf_lr = args.emb_learning_rate if 'crf_learning_rate' in dir(args) else 1.0
    bigru_num = args.bigru_num
    init_bound = 0.1
    IS_SPARSE = True

    def _bigru_layer(input_feature):
        """
        define the bidirectional gru layer
        """
        pre_gru = fluid.layers.fc(
            input=input_feature,
            size=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        gru = fluid.layers.dynamic_gru(
            input=pre_gru,
            size=grnn_hidden_dim,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        pre_gru_r = fluid.layers.fc(
            input=input_feature,
            size=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        gru_r = fluid.layers.dynamic_gru(
            input=pre_gru_r,
            size=grnn_hidden_dim,
            is_reverse=True,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        bi_merge = fluid.layers.concat(input=[gru, gru_r], axis=1)
        return bi_merge
    
    def log_softmax(logits, axis=-1):
        logsoftmax = logits-fluid.layers.log(fluid.layers.reduce_sum(fluid.layers.exp(logits),axis))
        return logsoftmax
   
    def cross_entropy(student, teacher):
        ce_loss = -1.0 * fluid.layers.reduce_sum(teacher*fluid.layers.log(student), dim=1)
        ce_loss = fluid.layers.sequence_pool(ce_loss, "sum")
        return ce_loss

    def kl_div(student, teacher):
        ce_loss = fluid.layers.reduce_sum(teacher*(fluid.layers.log(teacher) - fluid.layers.log(student)), dim=1)
        ce_loss = fluid.layers.sequence_pool(ce_loss, "sum")
        return ce_loss

    def pred(student, teacher,t=1.0):
        return fluid.layers.reduce_mean(-1.0*fluid.layers.softmax(teacher)*log_softmax(student/t))
   
    def normalize(alpha):
        """ alpha shape (-1, 57)
        """
        tag_num = alpha.shape[1] 
        sum_alpha = fluid.layers.reduce_sum(alpha, dim=1)
        sum_alpha = fluid.layers.unsqueeze(sum_alpha, axes=[1])
        sum_alpha = fluid.layers.expand(sum_alpha, [1, tag_num])
        norm_alpha = alpha / sum_alpha
        return norm_alpha
 
    def _net_conf(word, target=None):
        """
        Configure the network
        """
        word_embedding = fluid.embedding(
            input=word,
            size=[vocab_size, word_emb_dim],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr=fluid.ParamAttr(
                learning_rate=emb_lr,
                name="word_emb",
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound)))

        input_feature = word_embedding
        for i in range(bigru_num):
            bigru_output = _bigru_layer(input_feature)
            input_feature = bigru_output

        emission = fluid.layers.fc(
            size=num_labels,
            input=bigru_output,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        if target is not None:
            crf_cost = fluid.layers.linear_chain_crf(
                input=emission,
                label=target,
                param_attr=fluid.ParamAttr(
                    name='crfw', learning_rate=crf_lr))
            if teacher_crf_decode is not None:
                teacher_cost = pred(student=emission, teacher=teacher_crf_decode,t=1.0)
            else:
                teacher_cost = 0
                print('no teacher emission')
            crf_avg_cost = fluid.layers.mean(x=crf_cost)
            alpha, beta = 0.5, 0.5
            print("alpha * crf_avg_cost + beta * teacher_cost: ", alpha, beta)
            avg_cost = alpha * crf_avg_cost+ beta * teacher_cost
            crf_decode = fluid.layers.crf_decoding(
                input=emission, param_attr=fluid.ParamAttr(name='crfw'))
            return avg_cost, crf_avg_cost, teacher_cost, crf_decode

        else:
            size = emission.shape[1]
            fluid.layers.create_parameter(
                shape=[size + 2, size], dtype=emission.dtype, name='crfw')
            crf_decode = fluid.layers.crf_decoding(
                input=emission, param_attr=fluid.ParamAttr(name='crfw'))

        return crf_decode

    if for_infer:
        return _net_conf(word)

    else:
        # assert target != None, "target is necessary for training"
        return _net_conf(word, target)
