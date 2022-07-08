# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import numpy as np
import argparse
import paddle
import paddle.nn as nn
import functools
from functools import partial
import shutil
from paddle.io import Dataset, BatchSampler, DataLoader
from paddle.metric import Metric, Accuracy
from paddlenlp.transformers import AutoModelForTokenClassification, AutoTokenizer
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from paddleslim.auto_compression.config_helpers import load_config as load_slim_config
from paddleslim.auto_compression.compressor import AutoCompression


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='output',
        help="directory to save compressed model.")
    parser.add_argument(
        '--eval',
        type=bool,
        default=False,
        help="whether validate the model only.")
    return parser


METRIC_CLASSES = {
    "cola": Mcc,
    "sst-2": Accuracy,
    "mrpc": AccuracyAndF1,
    "sts-b": PearsonAndSpearman,
    "qqp": AccuracyAndF1,
    "mnli": Accuracy,
    "qnli": Accuracy,
    "rte": Accuracy,
}

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("sentence1", "sentence2"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("sentence1", "sentence2"),
    "qqp": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
    "sst-2": ("sentence", None),
    "sts-b": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False,
                    padding='max_length',
                    return_attention_mask=True):
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example['labels']
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    sentence1_key, sentence2_key = task_to_keys[global_config['task_name']]
    texts = ((example[sentence1_key], ) if sentence2_key is None else
             (example[sentence1_key], example[sentence2_key]))
    example = tokenizer(
        *texts,
        max_seq_len=max_seq_length,
        padding=padding,
        return_attention_mask=return_attention_mask,
        truncation='longest_first')
    if not is_test:
        if return_attention_mask:
            return example['input_ids'], example['attention_mask'], example[
                'token_type_ids'], label
        else:
            return example['input_ids'], example['token_type_ids'], label
    else:
        if return_attention_mask:
            return example['input_ids'], example['attention_mask'], example[
                'token_type_ids']
        else:
            return example['input_ids'], example['token_type_ids']


def create_data_holder(task_name, input_names):
    """
    Define the input data holder for the glue task.
    """
    inputs = []
    for name in input_names:
        inputs.append(
            paddle.static.data(
                name=name, shape=[-1, -1], dtype="int64"))

    if task_name == "sts-b":
        inputs.append(
            paddle.static.data(
                name="label", shape=[-1, 1], dtype="float32"))
    else:
        inputs.append(
            paddle.static.data(
                name="label", shape=[-1, 1], dtype="int64"))

    return inputs


def reader():
    # Create the tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(
        global_config['model_dir'], use_fast=False)
    train_ds = load_dataset(
        global_config['dataset'], global_config['task_name'], splits="train")

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=global_config['max_seq_length'],
        is_test=True,
        padding=global_config['padding'],
        return_attention_mask=global_config['return_attention_mask'])

    train_ds = train_ds.map(trans_func, lazy=True)
    if global_config['return_attention_mask']:
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=0),  # attention_mask
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type 
        ): fn(samples)
    else:
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type 
        ): fn(samples)

    train_batch_sampler = paddle.io.BatchSampler(
        train_ds,
        batch_size=global_config['batch_size'],
        shuffle=True,
        drop_last=True)

    feed_list = create_data_holder(global_config['task_name'],
                                   global_config['input_names'])
    train_data_loader = DataLoader(
        dataset=train_ds,
        feed_list=feed_list[:-1],
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=False)

    dev_trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=global_config['max_seq_length'],
        padding=global_config['padding'],
        return_attention_mask=global_config['return_attention_mask'])

    if global_config['return_attention_mask']:
        dev_batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=0),  # attention_mask
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type 
            Stack(dtype="int64" if train_ds.label_list else "float32")  # label
        ): fn(samples)
    else:
        dev_batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type 
            Stack(dtype="int64" if train_ds.label_list else "float32")  # label
        ): fn(samples)

    if global_config['task_name'] == "mnli":
        dev_ds_matched, dev_ds_mismatched = load_dataset(
            global_config['dataset'],
            global_config['task_name'],
            splits=["dev_matched", "dev_mismatched"])
        dev_ds_matched = dev_ds_matched.map(dev_trans_func, lazy=True)
        dev_ds_mismatched = dev_ds_mismatched.map(dev_trans_func, lazy=True)
        dev_batch_sampler_matched = paddle.io.BatchSampler(
            dev_ds_matched,
            batch_size=global_config['batch_size'],
            shuffle=False,
            drop_last=True)
        dev_data_loader_matched = DataLoader(
            dataset=dev_ds_matched,
            batch_sampler=dev_batch_sampler_matched,
            collate_fn=batchify_fn,
            feed_list=feed_list,
            num_workers=0,
            return_list=False)
        dev_batch_sampler_mismatched = paddle.io.BatchSampler(
            dev_ds_mismatched,
            batch_size=global_config['batch_size'],
            shuffle=False,
            drop_last=True)
        dev_data_loader_mismatched = DataLoader(
            dataset=dev_ds_mismatched,
            batch_sampler=dev_batch_sampler_mismatched,
            collate_fn=batchify_fn,
            num_workers=0,
            feed_list=feed_list,
            return_list=False,
            drop_last=True)
        return train_data_loader, dev_data_loader_matched, dev_data_loader_mismatched
    else:
        dev_ds = load_dataset(
            global_config['dataset'], global_config['task_name'], splits='dev')
        dev_ds = dev_ds.map(dev_trans_func, lazy=True)
        dev_batch_sampler = paddle.io.BatchSampler(
            dev_ds,
            batch_size=global_config['batch_size'],
            shuffle=False,
            drop_last=True)
        dev_data_loader = DataLoader(
            dataset=dev_ds,
            batch_sampler=dev_batch_sampler,
            collate_fn=dev_batchify_fn,
            num_workers=0,
            feed_list=feed_list,
            return_list=False)
        return train_data_loader, dev_data_loader


def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):
    metric.reset()
    for data in eval_dataloader():
        logits = exe.run(compiled_test_program,
                         feed={
                             test_feed_names[0]: data[0]['x0'],
                             test_feed_names[1]: data[0]['x1'],
                             test_feed_names[2]: data[0]['x2']
                         },
                         fetch_list=test_fetch_list)
        paddle.disable_static()
        if isinstance(metric, PearsonAndSpearman):
            labels_pd = paddle.to_tensor(np.array(data[0]['label'])).reshape(
                (-1, 1))
            logits_pd = paddle.to_tensor(logits[0]).reshape((-1, 1))
            metric.update((logits_pd, labels_pd))
        else:
            labels_pd = paddle.to_tensor(np.array(data[0]['label']).flatten())
            logits_pd = paddle.to_tensor(logits[0])
            correct = metric.compute(logits_pd, labels_pd)
            metric.update(correct)
        paddle.enable_static()
    res = metric.accumulate()
    return res[0] if isinstance(res, list) or isinstance(res, tuple) else res


def eval():
    devices = paddle.device.get_device().split(':')[0]
    places = paddle.device._convert_to_place(devices)
    exe = paddle.static.Executor(places)
    val_program, feed_target_names, fetch_targets = paddle.static.load_inference_model(
        global_config["model_dir"],
        exe,
        model_filename=global_config["model_filename"],
        params_filename=global_config["params_filename"])
    print('Loaded model from: {}'.format(global_config["model_dir"]))
    metric.reset()
    print('Evaluating...')
    for data in eval_dataloader():
        logits = exe.run(val_program,
                         feed={
                             feed_target_names[0]: data[0]['x0'],
                             feed_target_names[1]: data[0]['x1'],
                             feed_target_names[2]: data[0]['x2']
                         },
                         fetch_list=fetch_targets)
        paddle.disable_static()
        if isinstance(metric, PearsonAndSpearman):
            labels_pd = paddle.to_tensor(np.array(data[0]['label'])).reshape(
                (-1, 1))
            logits_pd = paddle.to_tensor(logits[0]).reshape((-1, 1))
            metric.update((logits_pd, labels_pd))
        else:
            labels_pd = paddle.to_tensor(np.array(data[0]['label']).flatten())
            logits_pd = paddle.to_tensor(logits[0])
            correct = metric.compute(logits_pd, labels_pd)
            metric.update(correct)
        paddle.enable_static()
    res = metric.accumulate()
    return res[0] if isinstance(res, list) or isinstance(res, tuple) else res


def apply_decay_param_fun(name):
    if name.find("bias") > -1:
        return True
    elif name.find("b_0") > -1:
        return True
    elif name.find("norm") > -1:
        return True
    else:
        return False


def main():
    all_config = load_slim_config(args.config_path)

    global global_config
    assert "Global" in all_config, "Key Global not found in config file."
    global_config = all_config["Global"]

    if 'TrainConfig' in all_config:
        all_config['TrainConfig']['optimizer_builder'][
            'apply_decay_param_fun'] = apply_decay_param_fun

    global train_dataloader, eval_dataloader
    train_dataloader, eval_dataloader = reader()

    global metric
    metric_class = METRIC_CLASSES[global_config['task_name']]
    metric = metric_class()

    if args.eval:
        result = eval()
        print('Eval metric:', result)
        sys.exit(0)

    ac = AutoCompression(
        model_dir=global_config['model_dir'],
        model_filename=global_config['model_filename'],
        params_filename=global_config['params_filename'],
        save_dir=args.save_dir,
        config=all_config,
        train_dataloader=train_dataloader,
        eval_callback=eval_function if
        (len(list(all_config.keys())) == 2 and 'TrainConfig' in all_config) or
        len(list(all_config.keys())) == 1 or
        'HyperParameterOptimization' not in all_config else eval_dataloader,
        eval_dataloader=eval_dataloader)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for file_name in os.listdir(global_config['model_dir']):
        if 'json' in file_name or 'txt' in file_name:
            shutil.copy(
                os.path.join(global_config['model_dir'], file_name),
                args.save_dir)

    ac.compress()


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    args = parser.parse_args()
    main()
