import os
import sys
import argparse
import functools
from functools import partial
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset, BatchSampler, DataLoader
from paddle.metric import Metric, Accuracy, Precision, Recall
# from paddlenlp.transformers import PPMiniLMForSequenceClassification, PPMiniLMTokenizer
# from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.transformers import AutoModelForTokenClassification, AutoTokenizer

from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.metrics import Mcc, PearsonAndSpearman
from paddleslim.auto_compression.config_helpers import load_config
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
    assert global_config['dataset'] in [
        'glue', 'clue'
    ], "This demo only supports for dataset glue or clue"
    """Convert a glue example into necessary features."""
    if global_config['dataset'] == 'glue':
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

    else:  #if global_config['dataset'] == 'clue':
        if not is_test:
            # `label_list == None` is for regression task
            label_dtype = "int64" if label_list else "float32"
            # Get the label
            example['label'] = np.array(
                example["label"], dtype="int64").reshape((-1, 1))
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
            text, query, pronoun, query_idx, pronoun_idx = example[
                'text'], example['target']['span1_text'], example['target'][
                    'span2_text'], example['target']['span1_index'], example[
                        'target']['span2_index']
            text_list = list(text)
            assert text[pronoun_idx:(pronoun_idx + len(
                pronoun))] == pronoun, "pronoun: {}".format(pronoun)
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

    tokenizer = AutoTokenizer.from_pretrained(global_config['model_dir'])

    train_ds, dev_ds = load_dataset(
        global_config['dataset'],
        global_config['task_name'],
        splits=('train', 'dev'))

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=global_config['max_seq_length'],
        is_test=True)

    train_ds = train_ds.map(trans_func, lazy=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type 
    ): fn(samples)

    train_batch_sampler = paddle.io.BatchSampler(
        train_ds, batch_size=global_config['batch_size'], shuffle=True)

    [input_ids, token_type_ids, labels] = create_data_holder(global_config[
        'task_name'])
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
        max_seq_length=global_config['max_seq_length'])
    dev_batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type 
        Stack(dtype="int64" if train_ds.label_list else "float32")  # label
    ): fn(samples)
    dev_ds = dev_ds.map(dev_trans_func, lazy=True)
    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=global_config['batch_size'], shuffle=False)
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


def eval():
    devices = paddle.device.get_device().split(':')[0]
    places = paddle.device._convert_to_place(devices)
    exe = paddle.static.Executor(places)
    val_program, feed_target_names, fetch_targets = paddle.static.load_inference_model(
        global_config['model_dir'],
        exe,
        model_filename=global_config['model_filename'],
        params_filename=global_config['params_filename'])
    print('Loaded model from: {}'.format(global_config['model_dir']))
    metric.reset()
    print('Evaluating...')
    for data in eval_dataloader():
        logits = exe.run(val_program,
                         feed={
                             feed_target_names[0]: data[0]['input_ids'],
                             feed_target_names[1]: data[0]['token_type_ids']
                         },
                         fetch_list=fetch_targets)
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


def main():

    all_config = load_config(args.config_path)

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
        eval_callback=eval_function
        if 'HyperParameterOptimization' not in all_config else eval_dataloader,
        eval_dataloader=eval_dataloader)

    ac.compress()


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    args = parser.parse_args()
    main()
