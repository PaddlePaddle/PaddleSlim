import os
import sys
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir)
import argparse
import functools
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset, BatchSampler, DataLoader
from paddle.metric import Metric, Accuracy, Precision, Recall
from paddlenlp.transformers import PPMiniLMForSequenceClassification, PPMiniLMTokenizer
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.metrics import Mcc, PearsonAndSpearman
from paddleslim.auto_compression.config_helpers import load_config
from paddleslim.auto_compression.compressor import AutoCompression
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('model_type',                  str,    None,         "model type can be bert or ppminilm.")
add_arg('model_dir',                   str,    None,         "inference model directory.")
add_arg('model_filename',              str,    None,         "inference model filename.")
add_arg('params_filename',             str,    None,         "inference params filename.")
add_arg('dataset',                     str,    None,         "datset name.")
add_arg('save_dir',                    str,    None,         "directory to save compressed model.")
add_arg('max_seq_length',              int,    128,          "max sequence length after tokenization.")
add_arg('batch_size',                  int,    1,            "train batch size.")
add_arg('task_name',                        str,    'sst-2',      "task name in glue.")
add_arg('config_path',                 str,    None,         "path of compression strategy config.")
# yapf: enable

METRIC_CLASSES = {
    "cola": Mcc,
    "sst-2": Accuracy,
    "sts-b": PearsonAndSpearman,
    "mnli": Accuracy,
    "qnli": Accuracy,
    "rte": Accuracy,
    "afqmc": Accuracy,
    "tnews": Accuracy,
    "iflytek": Accuracy,
    "ocnli": Accuracy,
    "cmnli": Accuracy,
    "cluewsc2020": Accuracy,
    "csl": Accuracy,
}


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    assert args.dataset in ['glue', 'clue'], "This demo only supports for dataset glue or clue"
    """Convert a glue example into necessary features."""
    if args.dataset == 'glue':
        if not is_test:
            # `label_list == None` is for regression task
            label_dtype = "int64" if label_list else "float32"
            # Get the label
            label = example['labels']
            label = np.array([label], dtype=label_dtype)
        # Convert raw text to feature
        example = tokenizer(example['sentence'], max_seq_len=max_seq_length)

        if not is_test:
            return example['input_ids'], example['token_type_ids'], label
        else:
            return example['input_ids'], example['token_type_ids']
        
    else: #if args.dataset == 'clue':
        if not is_test:
            # `label_list == None` is for regression task
            label_dtype = "int64" if label_list else "float32"
            # Get the label
            example['label'] = np.array(example["label"], dtype="int64").reshape((-1, 1))
            label = example['label']
        # Convert raw text to feature
        if 'keyword' in example:  # CSL
            sentence1 = " ".join(example['keyword'])
            example = {
                'sentence1': sentence1,
                'sentence2': example['abst'],
                'label': example['label']
            }
        elif 'target' in example:  # wsc
            text, query, pronoun, query_idx, pronoun_idx = example['text'], example[
                'target']['span1_text'], example['target']['span2_text'], example[
                    'target']['span1_index'], example['target']['span2_index']
            text_list = list(text)
            assert text[pronoun_idx:(pronoun_idx + len(pronoun)
                                     )] == pronoun, "pronoun: {}".format(pronoun)
            assert text[query_idx:(query_idx + len(query)
                                   )] == query, "query: {}".format(query)
            if pronoun_idx > query_idx:
                text_list.insert(query_idx, "_")
                text_list.insert(query_idx + len(query) + 1, "_")
                text_list.insert(pronoun_idx + 2, "[")
                text_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
            else:
                text_list.insert(pronoun_idx, "[")
                text_list.insert(pronoun_idx + len(pronoun) + 1, "]")
                text_list.insert(query_idx + 2, "_")
                text_list.insert(query_idx + len(query) + 2 + 1, "_")
            text = "".join(text_list)
            example['sentence'] = text
        if tokenizer is None:
            return example
        if 'sentence' in example:
            example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
        elif 'sentence1' in example:
            example = tokenizer(
                example['sentence1'],
                text_pair=example['sentence2'],
                max_seq_len=max_seq_length)
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
    if task_name == "sts-b":
        label = paddle.static.data(name="label", shape=[-1, 1], dtype="float32")
    else:
        label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")

    return [input_ids, token_type_ids, label]


def reader():
    # Create the tokenizer and dataset
    if args.model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    else: # ppminilm
        tokenizer = PPMiniLMTokenizer.from_pretrained(args.model_dir)
    train_ds, dev_ds = load_dataset(
        args.dataset, args.task_name, splits=('train', 'dev'))

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=args.max_seq_length,
        is_test=True)

    train_ds = train_ds.map(trans_func, lazy=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type 
    ): fn(samples)

    train_batch_sampler = paddle.io.BatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)

    [input_ids, token_type_ids, labels] = create_data_holder(args.task_name)
    feed_list_name = []
    train_data_loader = DataLoader(
        dataset=train_ds,
        feed_list=[input_ids, token_type_ids],
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=False)

    dev_trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=args.max_seq_length)
    dev_batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type 
        Stack(dtype="int64" if train_ds.label_list else "float32")  # label
    ): fn(samples)
    dev_ds = dev_ds.map(dev_trans_func, lazy=True)
    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=args.batch_size, shuffle=False)
    dev_data_loader = DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=dev_batchify_fn,
        num_workers=0,
        feed_list=[input_ids, token_type_ids, labels],
        return_list=False)

    return train_data_loader, dev_data_loader


def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):
    metric.reset()
    for data in eval_dataloader():
        logits = exe.run(compiled_test_program,
                         feed={
                             test_feed_names[0]: data[0]['input_ids'],
                             test_feed_names[1]: data[0]['token_type_ids']
                         },
                         fetch_list=test_fetch_list)
        paddle.disable_static()
        labels_pd = paddle.to_tensor(np.array(data[0]['label']).flatten())
        logits_pd = paddle.to_tensor(logits[0])
        correct = metric.compute(logits_pd, labels_pd)
        metric.update(correct)
        paddle.enable_static()
    res = metric.accumulate()
    return res


def apply_decay_param_fun(name):
    if name.find("bias") > -1:
        return True
    elif name.find("b_0") > -1:
        return True
    elif name.find("norm") > -1:
        return True
    else:
        return False


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    paddle.enable_static()

    compress_config, train_config = load_config(args.config_path)
    if train_config is not None and 'optim_args' in train_config:
        train_config['optim_args'][
            'apply_decay_param_fun'] = apply_decay_param_fun

    train_dataloader, eval_dataloader = reader()
    metric_class = METRIC_CLASSES[args.task_name]
    metric = metric_class()

    ac = AutoCompression(
        model_dir=args.model_dir,
        model_filename=args.model_filename,
        params_filename=args.params_filename,
        save_dir=args.save_dir,
        strategy_config=compress_config,
        train_config=train_config,
        train_dataloader=train_dataloader,
        eval_callback=eval_function
        if compress_config is None or 'HyperParameterOptimization' not in compress_config else
        eval_dataloader,
        eval_dataloader=eval_dataloader)

    ac.compress()
