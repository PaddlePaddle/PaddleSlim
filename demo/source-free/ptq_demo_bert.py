import os
import sys
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)

import numpy as np
from functools import partial
from collections import namedtuple, Iterable
import paddle
import paddle.nn as nn
from paddle.io import Dataset, BatchSampler, DataLoader
import imagenet_reader as reader
from paddleslim.source_free.auto_compression import AutoCompression
from paddleslim.source_free.strategy_config import QuantizationConfig, DistillationConfig, MultiTeacherDistillationConfig, HyperParameterOptimizationConfig, TrainConfig
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.data.sampler import SamplerHelper

paddle.enable_static()
default_ptq_config = {
   "quantize_op_types": ["conv2d", "depthwise_conv2d", "mul"],
   "weight_bits": 8,
   "activation_bits": 8,
}

default_hpo_config = {
"ptq_algo": ["KL", "hist"],
"bias_correct": [True],
"weight_quantize_type": ["channel_wise_abs_max"], 
"hist_percent": [0.999, 0.99999],
"batch_size": [4, 8, 16],
"batch_num": [4, 8, 16],
"max_quant_count": 20
}

def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    """
    Convert a glue example into necessary features.
    """
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example['labels']
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    #if (int(is_test) + len(example)) == 2:
    #    example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
    #else:
    #    example = tokenizer(
    #        example['sentence1'],
    #        text_pair=example['sentence2'],
    #        max_seq_len=max_seq_length)
    example = tokenizer(example['sentence'], max_seq_len=max_seq_length)

    if not is_test:
        return example['input_ids'], example['token_type_ids'], label
    else:
        return example['input_ids'], example['token_type_ids']


def create_data_holder(task_name):
    """
    Define the input data holder for the glue task.
    """
    input_ids = paddle.static.data(
        name="input_ids", shape=[-1, -1], dtype="int64")
    token_type_ids = paddle.static.data(
        name="token_type_ids", shape=[-1, -1], dtype="int64")
    #if task_name == "sts-b":
    #    label = paddle.static.data(name="label", shape=[-1, 1], dtype="float32")
    #else:
    #    label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")

    #return [input_ids, token_type_ids, label]
    return [input_ids, token_type_ids]

def reader():
    # Create the tokenizer and dataset
    tokenizer = BertTokenizer.from_pretrained('./static_bert_models/')
    train_ds = load_dataset('glue', 'sst-2', splits="train")

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=128,
        is_test=True)

    train_ds = train_ds.map(trans_func, lazy=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type 
        #Stack(dtype="int64" if train_ds.label_list else "float32")  # label
    ): fn(samples)

    train_batch_sampler = paddle.io.BatchSampler(
        train_ds, batch_size=32, shuffle=True)

    #[input_ids, token_type_ids, labels] = create_data_holder('sst-2')
    [input_ids, token_type_ids] = create_data_holder('sst-2')
    feed_list_name = []
    train_data_loader = DataLoader(
        dataset=train_ds,
        feed_list=[input_ids, token_type_ids],
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=False)

    dev_ds = load_dataset('glue', 'sst-2', splits='dev')
    dev_ds = dev_ds.map(trans_func, lazy=True)
    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=32, shuffle=False)
    dev_data_loader = DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        feed_list=[input_ids, token_type_ids],
        return_list=False)

    return train_data_loader, dev_data_loader

train_dataloader, eval_dataloader = reader()

ac = AutoCompression(model_dir='./static_bert_models', 
                     model_filename='bert.pdmodel', 
                     params_filename='bert.pdiparams', 
                     save_dir='./bert_ptq_hpo_output', 
                     strategy_config={"QuantizationConfig": QuantizationConfig(**default_ptq_config), 
                     "HyperParameterOptimizationConfig": HyperParameterOptimizationConfig(**default_hpo_config)}, 
                     train_config=None,
                     train_dataloader=train_dataloader, eval_callback=eval_dataloader)

ac.compression()
