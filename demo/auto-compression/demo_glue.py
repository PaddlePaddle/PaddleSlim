import os
import sys
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)
import argparse
import functools
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset, BatchSampler, DataLoader
from paddle.metric import Metric, Accuracy, Precision, Recall
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
add_arg('model_dir',                   str,    None,         "inference model directory.")
add_arg('model_filename',              str,    None,         "inference model filename.")
add_arg('params_filename',             str,    None,         "inference params filename.")
add_arg('save_dir',                    str,    None,         "directory to save compressed model.")
add_arg('devices',                     str,    'gpu',        "which device used to compress.")
add_arg('batch_size',                  int,    1,            "train batch size.")
add_arg('task',                        str,    'sst-2',      "task name in glue.")
add_arg('config_path',                 str,    None,         "path of compression strategy config.")
# yapf: enable

METRIC_CLASSES = {
    "cola": Mcc,
    "sst-2": Accuracy,
    "sts-b": PearsonAndSpearman,
    "mnli": Accuracy,
    "qnli": Accuracy,
    "rte": Accuracy,
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
    if task_name == "sts-b":
        label = paddle.static.data(name="label", shape=[-1, 1], dtype="float32")
    else:
        label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")

    return [input_ids, token_type_ids, label]


def reader():
    # Create the tokenizer and dataset
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    train_ds = load_dataset('glue', args.task, splits="train")

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
    ): fn(samples)

    train_batch_sampler = paddle.io.BatchSampler(
        train_ds, batch_size=32, shuffle=True)

    [input_ids, token_type_ids, labels] = create_data_holder(args.task)
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
        max_seq_length=128)
    dev_batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type 
        Stack(dtype="int64" if train_ds.label_list else "float32")  # label
    ): fn(samples)
    dev_ds = load_dataset('glue', args.task, splits='dev')
    dev_ds = dev_ds.map(dev_trans_func, lazy=True)
    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=32, shuffle=False)
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
        labels_pd = paddle.to_tensor(np.array(data[0]['label']))
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
    metric_class = METRIC_CLASSES[args.task]
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
        if 'HyperParameterOptimization' not in compress_config else
        eval_dataloader,
        eval_dataloader=eval_dataloader,
        devices=args.devices,
        model_type='transformer')

    ac.compress()
