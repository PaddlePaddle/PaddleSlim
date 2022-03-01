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
from paddleslim.source_free.auto_compression import AutoCompression
from paddleslim.source_free.strategy_config import *
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
add_arg('prune_algo',                  str,    'asp',        "prune algorithm.")
add_arg('distill_loss',                str,    'l2_loss',    "which loss to used in distillation.")
add_arg('distill_node_pair',           str,    None,         "distill node pair name list.", nargs="+")
add_arg('distill_lambda',              float,  1.0,          "weight of distill loss.")
add_arg('teacher_model_dir',           str,    None,         "teacher model directory.")
add_arg('teacher_model_filename',      str,    None,         "teacher model filename.")
add_arg('teacher_params_filename',     str,    None,         "teacher params filename.")
add_arg('epochs',                      int,    3,            "train epochs.")
add_arg('optimizer',                   str,    'SGD',        "optimizer to used.")
add_arg('learning_rate',               float,  0.0001,       "learning rate in optimizer.")
add_arg('eval_iter',                   int,    1000,         "how many iteration to eval.")
add_arg('origin_metric',               float,  None,         "metric of inference model to compressed.")
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
    ): fn(samples)

    train_batch_sampler = paddle.io.BatchSampler(
        train_ds, batch_size=32, shuffle=True)

    [input_ids, token_type_ids, labels] = create_data_holder('sst-2')
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
    dev_ds = load_dataset('glue', 'sst-2', splits='dev')
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


def eval_function(exe, place, compiled_test_program, test_feed_names,
                  test_fetch_list):
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


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    paddle.enable_static()

    default_prune_config = {"prune_algo": args.prune_algo}
    default_distill_config = {
        "distill_loss": args.distill_loss,
        "distill_node_pair": args.distill_node_pair,
        "distill_lambda": args.distill_lambda,
        "teacher_model_dir": args.teacher_model_dir,
        "teacher_model_filename": args.teacher_model_filename,
        "teacher_params_filename": args.teacher_params_filename,
    }

    def apply_decay_param_fun(name):
        if name.find("bias") > -1:
            return True
        elif name.find("norm") > -1:
            return True
        else:
            return False

    default_train_config = {
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "optim_args": {
            "apply_decay_param_fun": apply_decay_param_fun
        },
        "eval_iter": args.eval_iter,
        "origin_metric": args.origin_metric
    }

    train_dataloader, eval_dataloader = reader()
    metric_class = METRIC_CLASSES['sst-2']
    metric = metric_class()

    ac = AutoCompression(
        model_dir=args.model_dir,
        model_filename=args.model_filename,
        params_filename=args.params_filename,
        save_dir=args.save_dir,
        strategy_config={
            "PruneConfig": PruneConfig(**default_prune_config),
            "DistillationConfig": DistillationConfig(**default_distill_config)
        },
        train_config=TrainConfig(**default_train_config),
        train_dataloader=train_dataloader,
        eval_callback=eval_function,
        devices=args.devices)

    ac.compression()
