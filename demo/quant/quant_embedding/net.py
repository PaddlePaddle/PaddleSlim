# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
neural network for word2vec
"""
from __future__ import print_function
import math
import numpy as np
import paddle
import paddle.fluid as fluid


def skip_gram_word2vec(dict_size, embedding_size, is_sparse=False, neg_num=5):

    datas = []
    input_word = paddle.static.data(
        name="input_word", shape=[None, 1], dtype='int64')
    true_word = paddle.static.data(
        name='true_label', shape=[None, 1], dtype='int64')
    neg_word = paddle.static.data(
        name="neg_label", shape=[None, neg_num], dtype='int64')

    datas.append(input_word)
    datas.append(true_word)
    datas.append(neg_word)

    py_reader = fluid.layers.create_py_reader_by_data(
        capacity=64, feed_list=datas, name='py_reader', use_double_buffer=True)

    words = fluid.layers.read_file(py_reader)
    words[0] = paddle.reshape(words[0], [-1])
    words[1] = paddle.reshape(words[1], [-1])
    init_width = 0.5 / embedding_size
    input_emb = paddle.static.nn.embedding(
        input=words[0],
        is_sparse=is_sparse,
        size=[dict_size, embedding_size],
        param_attr=paddle.ParamAttr(
            name='emb',
            initializer=paddle.nn.initializer.Uniform(-init_width, init_width)))

    true_emb_w = paddle.static.nn.embedding(
        input=words[1],
        is_sparse=is_sparse,
        size=[dict_size, embedding_size],
        param_attr=paddle.ParamAttr(
            name='emb_w',
            initializer=paddle.nn.initializer.Constant(value=0.0)))

    true_emb_b = paddle.static.nn.embedding(
        input=words[1],
        is_sparse=is_sparse,
        size=[dict_size, 1],
        param_attr=paddle.ParamAttr(
            name='emb_b',
            initializer=paddle.nn.initializer.Constant(value=0.0)))
    neg_word_reshape = paddle.reshape(words[2], shape=[-1])
    neg_word_reshape.stop_gradient = True

    neg_emb_w = paddle.static.nn.embedding(
        input=neg_word_reshape,
        is_sparse=is_sparse,
        size=[dict_size, embedding_size],
        param_attr=paddle.ParamAttr(
            name='emb_w', learning_rate=1.0))

    neg_emb_w_re = paddle.reshape(
        neg_emb_w, shape=[-1, neg_num, embedding_size])
    neg_emb_b = paddle.static.nn.embedding(
        input=neg_word_reshape,
        is_sparse=is_sparse,
        size=[dict_size, 1],
        param_attr=paddle.ParamAttr(
            name='emb_b', learning_rate=1.0))

    neg_emb_b_vec = paddle.reshape(neg_emb_b, shape=[-1, neg_num])
    true_logits = paddle.add(paddle.mean(
        paddle.multiply(input_emb, true_emb_w), axis=1, keepdim=True),
                             true_emb_b)
    input_emb_re = paddle.reshape(input_emb, shape=[-1, 1, embedding_size])
    neg_matmul = fluid.layers.matmul(
        input_emb_re, neg_emb_w_re, transpose_y=True)
    neg_matmul_re = paddle.reshape(neg_matmul, shape=[-1, neg_num])
    neg_logits = paddle.add(neg_matmul_re, neg_emb_b_vec)
    #nce loss

    label_ones = fluid.layers.fill_constant_batch_size_like(
        true_logits, shape=[-1, 1], value=1.0, dtype='float32')
    label_zeros = fluid.layers.fill_constant_batch_size_like(
        true_logits, shape=[-1, neg_num], value=0.0, dtype='float32')

    true_xent = fluid.layers.sigmoid_cross_entropy_with_logits(true_logits,
                                                               label_ones)
    neg_xent = fluid.layers.sigmoid_cross_entropy_with_logits(neg_logits,
                                                              label_zeros)
    cost = paddle.add(paddle.sum(true_xent, axis=1),
                      paddle.sum(neg_xent, axis=1))
    avg_cost = paddle.mean(cost)
    return avg_cost, py_reader


def infer_network(vocab_size, emb_size):
    analogy_a = paddle.static.data(
        name="analogy_a", shape=[None, 1], dtype='int64')
    analogy_b = paddle.static.data(
        name="analogy_b", shape=[None, 1], dtype='int64')
    analogy_c = paddle.static.data(
        name="analogy_c", shape=[None, 1], dtype='int64')
    all_label = paddle.static.data(
        name="all_label", shape=[vocab_size, 1], dtype='int64')
    all_label = paddle.reshape(all_label, [-1])
    emb_all_label = paddle.static.nn.embedding(
        input=all_label, size=[vocab_size, emb_size], param_attr="emb")

    analogy_a = paddle.reshape(analogy_a, [-1])
    emb_a = paddle.static.nn.embedding(
        input=analogy_a, size=[vocab_size, emb_size], param_attr="emb")
    analogy_b = paddle.reshape(analogy_b, [-1])
    emb_b = paddle.static.nn.embedding(
        input=analogy_b, size=[vocab_size, emb_size], param_attr="emb")
    analogy_c = paddle.reshape(analogy_c, [-1])
    emb_c = paddle.static.nn.embedding(
        input=analogy_c, size=[vocab_size, emb_size], param_attr="emb")
    target = paddle.add(paddle.add(emb_b, -emb_a), emb_c)
    emb_all_label_l2 = fluid.layers.l2_normalize(x=emb_all_label, axis=1)
    dist = fluid.layers.matmul(x=target, y=emb_all_label_l2, transpose_y=True)
    values, pred_idx = paddle.topk(x=dist, k=4)
    return values, pred_idx
