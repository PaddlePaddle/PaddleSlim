import os
import sys
import argparse
import functools
from functools import partial
import numpy as np
import shutil
import paddle
import paddle.nn as nn
from paddle.io import Dataset, BatchSampler, DataLoader
from paddlenlp.transformers import AutoModelForTokenClassification, AutoTokenizer
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.metrics import SpanEvaluator

from paddleslim.common import load_config
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


def map_offset(ori_offset, offset_mapping):
    """
    map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1


def convert_example(example, tokenizer, max_seq_len, multilingual=True):
    """
    example: {
        title
        prompt
        content
        result_list
    }
    """
    encoded_inputs = tokenizer(
        text=[example["prompt"]],
        text_pair=[example["content"]],
        truncation=True,
        max_seq_len=max_seq_len,
        pad_to_max_seq_len=True,
        return_attention_mask=True,
        return_position_ids=True,
        return_dict=False,
        return_offsets_mapping=True)
    encoded_inputs = encoded_inputs[0]
    offset_mapping = [list(x) for x in encoded_inputs["offset_mapping"]]
    bias = 0
    for index in range(1, len(offset_mapping)):
        mapping = offset_mapping[index]
        if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
            bias = offset_mapping[index - 1][1] + 1  # Includes [SEP] token
        if mapping[0] == 0 and mapping[1] == 0:
            continue
        offset_mapping[index][0] += bias
        offset_mapping[index][1] += bias
    start_ids = [0.0 for x in range(max_seq_len)]
    end_ids = [0.0 for x in range(max_seq_len)]
    for item in example["result_list"]:
        start = map_offset(item["start"] + bias, offset_mapping)
        end = map_offset(item["end"] - 1 + bias, offset_mapping)
        start_ids[start] = 1.0
        end_ids[end] = 1.0
    if multilingual:
        tokenized_output = {
            "input_ids": encoded_inputs["input_ids"],
            "pos_ids": encoded_inputs["position_ids"],
            "start_positions": start_ids,
            "end_positions": end_ids
        }
    else:
        tokenized_output = {
            "input_ids": encoded_inputs["input_ids"],
            "token_type_ids": encoded_inputs["token_type_ids"],
            "pos_ids": encoded_inputs["position_ids"],
            "att_mask": encoded_inputs["attention_mask"],
            "start_positions": start_ids,
            "end_positions": end_ids
        }
    return tokenized_output


def create_data_holder(multilingual=True):
    """
    Define the input data holder for the glue task.
    """

    input_ids = paddle.static.data(
        name="input_ids", shape=[-1, -1], dtype="int64")
    return_lis = [input_ids]

    token_type_ids = paddle.static.data(
        name="token_type_ids", shape=[-1, -1], dtype="int64")
    return_list.append(token_type_ids)

    if not multilingual:
        position_ids = paddle.static.data(
            name="pos_ids", shape=[-1, -1], dtype="int64")
        attention_mask = paddle.static.data(
            name="att_mask", shape=[-1, -1], dtype="int64")
        return_list.append(position_ids)
        return_list.append(attention_mask)

    start_ids = paddle.static.data(
        name="start_ids", shape=[-1, 1], dtype="float32")
    end_ids = paddle.static.data(name="end_ids", shape=[-1, 1], dtype="float32")
    return_list.append(start_ids)
    return_list.append(end_ids)

    return return_list


def reader():
    train_ds = load_dataset(
        reader,
        data_path=global_config['train_data'],
        max_seq_len=global_config['max_seq_length'],
        lazy=False)
    dev_ds = load_dataset(
        reader,
        data_path=global_config['dev_data'],
        max_seq_len=global_config['max_seq_length'],
        lazy=False)

    tokenizer = AutoTokenizer.from_pretrained(global_config['model_dir'])

    trans_fn = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_len=global_config['max_seq_length'],
        is_test=True)
    train_ds = train_ds.map(trans_fn)

    dev_trans_fn = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_len=global_config['max_seq_length'])
    dev_ds = dev_ds.map(dev_trans_fn)

    batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    }): fn(samples)

    dev_batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        'start_ids': Stack(dtype="int64"),
        'end_ids': Stack(dtype="int64")}): fn(samples)

    [input_ids, token_type_ids, start_ids, end_ids] = create_data_holder()

    train_batch_sampler = paddle.io.BatchSampler(
        dataset=train_ds, batch_size=global_config['batch_size'], shuffle=True)
    train_dataloader = paddle.io.DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        return_list=False,
        feed_list=[input_ids, token_type_ids, start_ids, end_ids],
        collate_fn=batchify_fn)

    dev_batch_sampler = paddle.io.BatchSampler(
        dataset=dev_ds, batch_size=global_config['batch_size'], shuffle=False)
    eval_dataloader = paddle.io.DataLoader(
        dev_ds,
        batch_sampler=dev_batch_sampler,
        return_list=False,
        feed_list=[input_ids, token_type_ids],
        collate_fn=dev_batchify_fn)
    return train_dataloader, eval_dataloader


def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):
    metric.reset()
    for data in eval_dataloader():
        logits = exe.run(compiled_test_program,
                         feed={
                             'input_ids': data[0]['input_ids'],
                             'token_type_ids': data[0]['token_type_ids'],
                             'pos_ids': data[0]['pos_ids'],
                             'att_mask': data[0]['att_mask'],
                         },
                         fetch_list=test_fetch_list)
        paddle.disable_static()

        start_ids = paddle.to_tensor(np.array(data[0]['start_ids']))
        end_ids = paddle.to_tensor(np.array(data[0]['end_ids']))

        start_prob = paddle.to_tensor(logits[0])
        end_prob = paddle.to_tensor(logits[1])

        num_correct, num_infer, num_label = metric.compute(start_prob, end_prob,
                                                           start_ids, end_ids)
        metric.update(num_correct, num_infer, num_label)
        paddle.enable_static()
    precision, recall, f1 = metric.accumulate()
    return f1


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
    metric = SpanEvaluator()

    ac = AutoCompression(
        model_dir=global_config['model_dir'],
        model_filename=global_config['model_filename'],
        params_filename=global_config['params_filename'],
        save_dir=args.save_dir,
        config=all_config,
        train_dataloader=train_dataloader,
        eval_callback=eval_function,
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
