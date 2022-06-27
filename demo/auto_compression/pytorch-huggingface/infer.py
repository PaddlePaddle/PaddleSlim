# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os
import time
import sys
from functools import partial
import distutils.util
import numpy as np

import paddle
from paddle import inference
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddle.metric import Metric, Accuracy, Precision, Recall
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer

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


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default='cola',
        type=str,
        help="The name of the task to perform predict, selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()), )
    parser.add_argument(
        "--model_type",
        default='bert-base-cased',
        type=str,
        help="Model type selected in bert.")
    parser.add_argument(
        "--model_name_or_path",
        default='bert-base-cased',
        type=str,
        help="The directory or name of model.", )
    parser.add_argument(
        "--model_path",
        default='./quant_models/model',
        type=str,
        required=True,
        help="The path prefix of inference model to be used.", )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu", "xpu"],
        help="Device selected for inference.", )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for predict.", )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--padding",
        default='max_length',
        type=int,
        help="Padding type", )
    parser.add_argument(
        "--perf_warmup_steps",
        default=20,
        type=int,
        help="Warmup steps for performance test.", )
    parser.add_argument(
        "--use_trt",
        action='store_true',
        help="Whether to use inference engin TensorRT.", )
    parser.add_argument(
        "--perf",
        action='store_false',
        help="Whether to test performance.", )
    parser.add_argument(
        "--int8",
        action='store_true',
        help="Whether to use int8 inference.", )

    parser.add_argument(
        "--fp16",
        action='store_true',
        help="Whether to use int8 inference.", )
    args = parser.parse_args()
    return args


@paddle.no_grad()
def evaluate(outputs, metric, data_loader):
    metric.reset()
    for i, batch in enumerate(data_loader):
        input_ids, segment_ids, labels = batch
        logits = paddle.to_tensor(outputs[i][0])
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    print("acc: %s, " % res, end='')


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    task_name=None,
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
    sentence1_key, sentence2_key = task_to_keys[task_name]
    texts = ((example[sentence1_key], ) if sentence2_key is None else
             (example[sentence1_key], example[sentence2_key]))
    example = tokenizer(
        *texts,
        max_seq_len=max_seq_length,
        padding=padding,
        return_attention_mask=return_attention_mask)
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


class Predictor(object):
    def __init__(self, predictor, input_handles, output_handles):
        self.predictor = predictor
        self.input_handles = input_handles
        self.output_handles = output_handles

    @classmethod
    def create_predictor(cls, args):
        config = paddle.inference.Config(args.model_path + ".pdmodel",
                                         args.model_path + ".pdiparams")
        if args.device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
            cls.device = paddle.set_device("gpu")
        elif args.device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
            cls.device = paddle.set_device("cpu")
        elif args.device == "xpu":
            # set XPU configs accordingly
            config.enable_xpu(100)
        if args.use_trt:
            if args.int8:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=inference.PrecisionType.Int8,
                    max_batch_size=args.batch_size,
                    min_subgraph_size=5,
                    use_static=False,
                    use_calib_mode=False)
            elif args.fp16:

                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=inference.PrecisionType.Half,
                    max_batch_size=args.batch_size,
                    min_subgraph_size=5,
                    use_static=False,
                    use_calib_mode=False)
            else:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=inference.PrecisionType.Float32,
                    max_batch_size=args.batch_size,
                    min_subgraph_size=5,
                    use_static=False,
                    use_calib_mode=False)
            print("Enable TensorRT is: {}".format(
                config.tensorrt_engine_enabled()))
            # Set min/max/opt tensor shape of each trt subgraph input according
            # to dataset.
            # For example, the config of TNEWS data should be 16, 32, 32, 31, 128, 32.
        predictor = paddle.inference.create_predictor(config)

        input_handles = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        output_handles = [
            predictor.get_output_handle(name)
            for name in predictor.get_output_names()
        ]

        return cls(predictor, input_handles, output_handles)

    def predict_batch(self, data):
        for input_field, input_handle in zip(data, self.input_handles):
            input_handle.copy_from_cpu(input_field)
        self.predictor.run()
        output = [
            output_handle.copy_to_cpu() for output_handle in self.output_handles
        ]
        return output

    def convert_predict_batch(self, args, data, tokenizer, batchify_fn,
                              label_list):
        examples = []
        for example in data:
            example = convert_example(
                example,
                tokenizer,
                label_list,
                task_name=args.task_name,
                max_seq_length=args.max_seq_length,
                padding='max_length',
                return_attention_mask=True)
            examples.append(example)

        return examples

    def predict(self, dataset, tokenizer, batchify_fn, args):
        batches = [
            dataset[idx:idx + args.batch_size]
            for idx in range(0, len(dataset), args.batch_size)
        ]
        if args.perf:
            for i, batch in enumerate(batches):
                examples = self.convert_predict_batch(
                    args, batch, tokenizer, batchify_fn, dataset.label_list)
                input_ids, atten_mask, segment_ids, label = batchify_fn(
                    examples)
                output = self.predict_batch(
                    [input_ids, atten_mask, segment_ids])
                if i > args.perf_warmup_steps:
                    break
            time1 = time.time()
            for batch in batches:
                examples = self.convert_predict_batch(
                    args, batch, tokenizer, batchify_fn, dataset.label_list)
                input_ids, atten_mask, segment_ids, _ = batchify_fn(examples)
                output = self.predict_batch(
                    [input_ids, atten_mask, segment_ids])

            print("task name: %s, time: %s, " %
                  (args.task_name, time.time() - time1))

        else:
            metric = METRIC_CLASSES[args.task_name]()
            metric.reset()
            for i, batch in enumerate(batches):
                examples = self.convert_predict_batch(
                    args, batch, tokenizer, batchify_fn, dataset.label_list)
                input_ids, atten_mask, segment_ids, label = batchify_fn(
                    examples)
                output = self.predict_batch(
                    [input_ids, atten_mask, segment_ids])
                correct = metric.compute(
                    paddle.to_tensor(output), paddle.to_tensor(label))
                metric.update(correct)

            res = metric.accumulate()
            print("task name: %s, acc: %s, " % (args.task_name, res), end='')


def main():
    paddle.seed(42)
    args = parse_args()

    args.task_name = args.task_name.lower()
    args.model_type = args.model_type.lower()

    predictor = Predictor.create_predictor(args)

    dev_ds = load_dataset('glue', args.task_name, splits='dev')

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=0),
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        Stack(dtype="int64" if dev_ds.label_list else "float32")  # label
    ): fn(samples)
    outputs = predictor.predict(dev_ds, tokenizer, batchify_fn, args)


if __name__ == "__main__":
    main()
