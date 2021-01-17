# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import logging
import os
import random
import time
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
from paddle.metric import Accuracy

from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import BertModel, BertForSequenceClassification, BertTokenizer
from paddlenlp.utils.log import logger
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
import paddlenlp.datasets as datasets
from paddleslim.nas.ofa import OFA, DistillConfig, utils
from paddleslim.nas.ofa.convert_super import Convert, supernet

TASK_CLASSES = {
    "cola": (datasets.GlueCoLA, Mcc),
    "sst-2": (datasets.GlueSST2, Accuracy),
    "mrpc": (datasets.GlueMRPC, AccuracyAndF1),
    "sts-b": (datasets.GlueSTSB, PearsonAndSpearman),
    "qqp": (datasets.GlueQQP, AccuracyAndF1),
    "mnli": (datasets.GlueMNLI, Accuracy),
    "qnli": (datasets.GlueQNLI, Accuracy),
    "rte": (datasets.GlueRTE, Accuracy),
}

MODEL_CLASSES = {"bert": (BertForSequenceClassification, BertTokenizer), }


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(TASK_CLASSES.keys()), )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--sub_model_output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the sub model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--static_sub_model",
        default=None,
        type=str,
        required=True,
        help="The output directory where the sub static model will be written. If set to None, not export static model",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="number of gpus to use, 0 for cpu.")
    parser.add_argument(
        '--width_mult',
        type=float,
        default=1.0,
        help="width mult you want to export")
    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed + paddle.distributed.get_rank())
    np.random.seed(args.seed + paddle.distributed.get_rank())
    paddle.seed(args.seed + paddle.distributed.get_rank())


#### monkey patch for bert forward to accept [attention_mask, head_mask] as  attention_mask
def bert_forward(self,
                 input_ids,
                 token_type_ids=None,
                 position_ids=None,
                 attention_mask=[None, None]):
    wtype = self.pooler.dense.fn.weight.dtype if hasattr(
        self.pooler.dense, 'fn') else self.pooler.dense.weight.dtype
    if attention_mask[0] is None:
        attention_mask[0] = paddle.unsqueeze(
            (input_ids == self.pad_token_id).astype(wtype) * -1e9, axis=[1, 2])
    embedding_output = self.embeddings(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids)
    encoder_outputs = self.encoder(embedding_output, attention_mask)
    sequence_output = encoder_outputs
    pooled_output = self.pooler(sequence_output)
    return sequence_output, pooled_output


BertModel.forward = bert_forward


def sequence_forward(self,
                     input_ids,
                     token_type_ids=None,
                     position_ids=None,
                     attention_mask=[None, None]):
    _, pooled_output = self.bert(
        input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask)

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    return logits


BertForSequenceClassification.forward = sequence_forward


def sub_model_config(model, width_mult):
    new_config = dict()
    block_num = np.floor((len(model.layers.items()) - 3) / 6)
    block_name = block_num * 6 + 2

    def fix_exp(idx):
        if (idx - 3) % 6 == 0 or (idx - 5) % 6 == 0:
            return True
        return False

    start_idx = 0
    for idx, (block_k, block_v) in enumerate(model.layers.items()):
        if 'linear' in block_k:
            start_idx = int(block_k.split('_')[1])
            break

    for idx, (block_k, block_v) in enumerate(model.layers.items()):
        if isinstance(block_v, dict) and len(block_v.keys()) != 0:
            name, name_idx = block_k.split('_'), int(block_k.split('_')[1])
            if 'emb' in block_k or idx > block_name:
                new_config[block_k] = [1.0, 1.0]
            else:
                if fix_exp(name_idx - start_idx):
                    new_config[block_k] = [width_mult, 1.0]
                else:
                    new_config[block_k] = [1.0, width_mult]

    return new_config


def export_model(model, super_model, sub_model_config):
    def split_prefix(net, name_list):
        if len(name_list) > 1:
            net = split_prefix(getattr(net, name_list[0]), name_list[1:])
        elif len(name_list) == 1:
            net = getattr(net, name_list[0])
        else:
            raise NotImplementedError("name error")
        return net

    pre_exp = 1.0
    list_name = []
    for name, _ in model.named_sublayers():
        list_name.append(name)
    for name, sublayer in super_model.named_sublayers():
        if name not in list_name:
            continue
        sublayer1 = split_prefix(model, name.split('.'))
        if hasattr(sublayer, 'fn'):
            sublayer = sublayer.fn
        for param1, param2 in zip(sublayer1.parameters(include_sublayers=False),
                                  sublayer.parameters(include_sublayers=False)):
            t_value = param1.value().get_tensor()
            value = np.array(t_value).astype("float32")

            name = param2.name.split('.')[0][6:]
            if name in sub_model_config.keys():
                in_exp = sub_model_config[name][0]
                out_exp = sub_model_config[name][1]
                pre_exp = out_exp
                if len(value.shape) == 2:
                    in_chn, out_chn = int(value.shape[0] * in_exp), int(
                        value.shape[1] * out_exp)
                    prune_value = value[:in_chn, :out_chn]
                else:
                    out_chn = int(value.shape[0] * out_exp)
                    prune_value = value[:out_chn]
            else:
                out_chn = int(pre_exp * value.shape[0])
                prune_value = value[:out_chn]

            p = t_value._place()
            if p.is_cpu_place():
                place = paddle.CPUPlace()
            elif p.is_cuda_pinned_place():
                place = paddle.CUDAPinnedPlace()
            else:
                place = paddle.CUDAPlace(p.gpu_device_id())
            t_value.set(prune_value, place)


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    """convert a glue example into necessary features"""

    def _truncate_seqs(seqs, max_seq_length):
        if len(seqs) == 1:  # single sentence
            # Account for [CLS] and [SEP] with "- 2"
            seqs[0] = seqs[0][0:(max_seq_length - 2)]
        else:  # sentence pair
            # Account for [CLS], [SEP], [SEP] with "- 3"
            tokens_a, tokens_b = seqs
            max_seq_length -= 3
            while True:  # truncate with longest_first strategy
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_seq_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()
        return seqs

    def _concat_seqs(seqs, separators, seq_mask=0, separator_mask=1):
        concat = sum((seq + sep for sep, seq in zip(separators, seqs)), [])
        segment_ids = sum(([i] * (len(seq) + len(sep)) for i, (sep, seq) in
                           enumerate(zip(separators, seqs))), [])
        if isinstance(seq_mask, int):
            seq_mask = [[seq_mask] * len(seq) for seq in seqs]
        if isinstance(separator_mask, int):
            separator_mask = [[separator_mask] * len(sep) for sep in separators]
        p_mask = sum((s_mask + mask for sep, seq, s_mask, mask in
                      zip(separators, seqs, seq_mask, separator_mask)), [])
        return concat, segment_ids, p_mask

    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # get the label
        label = example[-1]
        example = example[:-1]
        #create label maps if classification task
        if label_list:
            label_map = {}
            for (i, l) in enumerate(label_list):
                label_map[l] = i
            label = label_map[label]
        label = np.array([label], dtype=label_dtype)

    # tokenize raw text
    tokens_raw = [tokenizer(l) for l in example]
    # truncate to the truncate_length,
    tokens_trun = _truncate_seqs(tokens_raw, max_seq_length)
    # concate the sequences with special tokens
    tokens_trun[0] = [tokenizer.cls_token] + tokens_trun[0]
    tokens, segment_ids, _ = _concat_seqs(tokens_trun, [[tokenizer.sep_token]] *
                                          len(tokens_trun))
    # convert the token to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    valid_length = len(input_ids)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    # input_mask = [1] * len(input_ids)
    if not is_test:
        return input_ids, segment_ids, valid_length, label
    else:
        return input_ids, segment_ids, valid_length


def export_static_model(model, model_path):
    input_shape = [
        paddle.static.InputSpec(
            shape=[None, 1], dtype='int64'), paddle.static.InputSpec(
                shape=[None, 1], dtype='int64')
    ]
    net = paddle.jit.to_static(model, input_spec=input_shape)
    paddle.jit.save(net, model_path)


def do_train(args):
    paddle.set_device("gpu" if args.n_gpu else "cpu")
    args.task_name = args.task_name.lower()
    dataset_class, metric_class = TASK_CLASSES[args.task_name]
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    train_ds = dataset_class.get_datasets(['train'])

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.get_labels(),
        max_seq_length=args.max_seq_length)
    train_ds = train_ds.apply(trans_func, lazy=True)

    num_labels = 1 if train_ds.get_labels() == None else len(
        train_ds.get_labels())

    model = model_class.from_pretrained(
        args.model_name_or_path, num_classes=num_labels)

    sp_config = supernet(expand_ratio=[1.0, args.width_mult])
    model = Convert(sp_config).convert(model)
    weights = os.path.join(args.model_name_or_path, 'model_state.pdparams')
    origin_weights = paddle.load(weights)
    model.set_state_dict(origin_weights)
    print("init supernet model done")

    ofa_model = OFA(model)
    sub_model_cfg = sub_model_config(ofa_model, args.width_mult)

    origin_model = model_class.from_pretrained(
        args.model_name_or_path, num_classes=num_labels)
    new_dict = utils.utils.remove_model_fn(origin_model, origin_weights)
    origin_model.set_state_dict(new_dict)
    print("init origin model done")

    export_model(origin_model, ofa_model.model, sub_model_cfg)

    for name, sublayer in origin_model.named_sublayers():
        if isinstance(sublayer, paddle.nn.MultiHeadAttention):
            sublayer.num_heads = int(args.width_mult * sublayer.num_heads)

    if args.static_sub_model != None:
        export_static_model(origin_model, args.static_sub_model)

    output_dir = os.path.join(args.sub_model_output_dir,
                              "model_width_%.5f" % args.width_mult)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # need better way to get inner model of DataParallel
    model_to_save = origin_model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    if args.n_gpu > 1:
        paddle.distributed.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
    else:
        do_train(args)
