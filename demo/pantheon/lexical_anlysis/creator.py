# -*- coding: UTF-8 -*-
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
"""
Define the function to create lexical analysis model and model's data reader
"""
import sys
import os
import math
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import NormalInitializer

from reader import Dataset
from ernie_reader import SequenceLabelReader

from models.sequence_labeling import nets
from models.representation.ernie import ernie_encoder, ernie_pyreader


def create_model(args, vocab_size, num_labels, mode='train'):
    """create lac model"""

    # model's input data
    words = fluid.data(name='words', shape=[-1, 1], dtype='int64', lod_level=1)
    targets = fluid.data(
        name='targets', shape=[-1, 1], dtype='int64', lod_level=1)
    if mode == "train":
        print("create model mode: ", mode)
        teacher_crf_decode = fluid.data(
            name='teacher_crf_decode', shape=[-1, 1], dtype='float32', lod_level=1)
    else:
        print("create model mode: ", mode)
        teacher_crf_decode = None
    
    feed_list = [words, targets]
    if teacher_crf_decode:
        feed_list.append(teacher_crf_decode)
        
    pyreader = fluid.io.DataLoader.from_generator(
                feed_list=feed_list,
                capacity=200,
                use_double_buffer=True,
                iterable=False)
    # for test or train process
    avg_cost, crf_avg_cost, teacher_cost, crf_decode= nets.lex_net(
        words, args, vocab_size, num_labels, teacher_crf_decode,for_infer=False, target=targets)

    (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
     num_correct_chunks) = fluid.layers.chunk_eval(
         input=crf_decode,
         label=targets,
         chunk_scheme="IOB",
         num_chunk_types=int(math.ceil((num_labels - 1) / 2.0)))
    chunk_evaluator = fluid.metrics.ChunkEvaluator()
    chunk_evaluator.reset()
    
    ret = {
        "pyreader": pyreader,
        "words": words,
        "targets": targets,
        "avg_cost": avg_cost,
        "crf_avg_cost": crf_avg_cost,
        "teacher_cost": teacher_cost,
        "crf_decode": crf_decode,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "chunk_evaluator": chunk_evaluator,
        "num_infer_chunks": num_infer_chunks,
        "num_label_chunks": num_label_chunks,
        "num_correct_chunks": num_correct_chunks
    }
    return ret

def create_lexnet_data_generator(args,
                                 reader,
                                 file_name,
                                 place,
                                 mode='train'):
    if mode == 'train':
        def wrapper():
            batch_words, batch_labels, batch_emissions, seq_lens = [], [], None, []
            emi_lens = []
            for epoch in range(args.epoch):
                print("data epoch: {}".format(epoch))
                for instance in reader.file_reader(file_name, mode="train")():
                    words, labels, emission = instance
                    if len(seq_lens) < args.batch_size:
                        batch_words.append(words)
                        batch_labels.append(labels)
                        if batch_emissions is not None:
                            batch_emissions = np.concatenate((batch_emissions, emission))
                        else:
                            batch_emissions = emission
                        seq_lens.append(len(words))
                        emi_lens.append(emission.shape[0])
                    if len(seq_lens) == args.batch_size:  
   
                        #print("batch words len", [len(seq) for seq in batch_words])
                        #print("batch labels len", [len(seq) for seq in batch_labels])
                        #print("emi lens:", emi_lens)
                        #print("emission first dim:", batch_emissions.shape[0])
                        #print("reduced seq_lens:", sum(seq_lens))
                        t_words = fluid.create_lod_tensor(batch_words, [seq_lens], place)
                        t_labels = fluid.create_lod_tensor(batch_labels, [seq_lens], place)
                        t_emissions = fluid.create_lod_tensor(batch_emissions, [seq_lens], place)
                        yield t_words, t_labels, t_emissions
                        batch_words, batch_labels, batch_emissions, seq_lens = [], [], None, []
                        emi_lens = []

                if len(seq_lens) > 0:                
                    t_words = fluid.create_lod_tensor(batch_words, [seq_lens], place)
                    t_labels = fluid.create_lod_tensor(batch_labels, [seq_lens], place)
                    t_emissions = fluid.create_lod_tensor(batch_emissions, [seq_lens], place)
                    yield t_words, t_labels, t_emissions
                    batch_words, batch_labels, batch_emissions, seq_lens = [], [], None, []

    else:
        def wrapper():
            batch_words, batch_labels, seq_lens = [], [], []
            for instance in reader.file_reader(file_name, mode="test")():
                words, labels = instance
                if len(seq_lens) < args.batch_size:
                    batch_words.append(words)
                    batch_labels.append(labels)
                    seq_lens.append(len(words))
                if len(seq_lens) == args.batch_size:  
                    t_words = fluid.create_lod_tensor(batch_words, [seq_lens], place)
                    t_labels = fluid.create_lod_tensor(batch_labels, [seq_lens], place)
                    yield t_words, t_labels
                    batch_words, batch_labels, seq_lens = [], [], []
    
            if len(seq_lens) > 0:                
                t_words = fluid.create_lod_tensor(batch_words, [seq_lens], place)
                t_labels = fluid.create_lod_tensor(batch_labels, [seq_lens], place)
                yield t_words, t_labels
                batch_words, batch_labels, seq_lens = [], [], []
    return wrapper

def create_pyreader(args,
                    file_name,
                    feed_list,
                    place,
                    model='lac',
                    reader=None,
                    return_reader=False,
                    mode='train'):
    reader = SequenceLabelReader(
                vocab_path=args.vocab_path,
                label_map_config=args.label_map_config,
                max_seq_len=args.max_seq_len,
                do_lower_case=args.do_lower_case,
                random_seed=args.random_seed)
    return reader.data_generator(file_name,args.batch_size,args.epoch,shuffle=False,phase="train")


def create_ernie_model(args, ernie_config):
    """
    Create Model for LAC based on ERNIE encoder
    """
    # ERNIE's input data

    src_ids = fluid.data(
        name='src_ids', shape=[-1, args.max_seq_len, 1], dtype='int64')
    sent_ids = fluid.data(
        name='sent_ids', shape=[-1, args.max_seq_len, 1], dtype='int64')
    pos_ids = fluid.data(
        name='pos_ids', shape=[-1, args.max_seq_len, 1], dtype='int64')
    input_mask = fluid.data(
        name='input_mask', shape=[-1, args.max_seq_len, 1], dtype='float32')

    padded_labels = fluid.data(
        name='padded_labels', shape=[-1, args.max_seq_len, 1], dtype='int64')

    seq_lens = fluid.data(
        name='seq_lens', shape=[-1], dtype='int64', lod_level=0)

    squeeze_labels = fluid.layers.squeeze(padded_labels, axes=[-1])

    # ernie_pyreader
    ernie_inputs = {
        "src_ids": src_ids,
        "sent_ids": sent_ids,
        "pos_ids": pos_ids,
        "input_mask": input_mask,
        "seq_lens": seq_lens
    }
    embeddings = ernie_encoder(ernie_inputs, ernie_config=ernie_config)

    padded_token_embeddings = embeddings["padded_token_embeddings"]

    emission = fluid.layers.fc(
        size=args.num_labels,
        input=padded_token_embeddings,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
                low=-args.init_bound, high=args.init_bound),
            regularizer=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=1e-4)),
        num_flatten_dims=2)

    crf_cost = fluid.layers.linear_chain_crf(
        input=emission,
        label=padded_labels,
        param_attr=fluid.ParamAttr(
            name='crfw', learning_rate=args.crf_learning_rate),
        length=seq_lens)

    avg_cost = fluid.layers.mean(x=crf_cost)
    crf_decode = fluid.layers.crf_decoding(
        input=emission,
        param_attr=fluid.ParamAttr(name='crfw'),
        length=seq_lens)

    (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
     num_correct_chunks) = fluid.layers.chunk_eval(
         input=crf_decode,
         label=squeeze_labels,
         chunk_scheme="IOB",
         num_chunk_types=int(math.ceil((args.num_labels - 1) / 2.0)),
         seq_length=seq_lens)
    chunk_evaluator = fluid.metrics.ChunkEvaluator()
    chunk_evaluator.reset()

    ret = {
        "feed_list":
        [src_ids, sent_ids, pos_ids, input_mask, padded_labels, seq_lens],
        "words": src_ids,
        "pos_ids":pos_ids,
        "sent_ids":sent_ids,
        "input_mask":input_mask,
        "labels": padded_labels,
        "seq_lens": seq_lens,
        "avg_cost": avg_cost,
        "crf_decode": crf_decode,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "chunk_evaluator": chunk_evaluator,
        "num_infer_chunks": num_infer_chunks,
        "num_label_chunks": num_label_chunks,
        "num_correct_chunks": num_correct_chunks,
        "emission":emission, 
        "alpha": None
    }

    return ret
