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
from paddle.metric import Metric, Accuracy, Precision, Recall
from paddlenlp.transformers import AutoModelForTokenClassification, AutoTokenizer
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.metrics import Mcc, PearsonAndSpearman

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


def convert_example(example, dataset, tokenizer, label_list,
                    max_seq_length=512):
    assert dataset in ['glue', 'clue'
                       ], "This demo only supports for dataset glue or clue"
    """Convert a glue example into necessary features."""
    if dataset == 'glue':
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example['labels']
        label = np.array([label], dtype=label_dtype)
        # Convert raw text to feature
        example = tokenizer(example['sentence'], max_seq_len=max_seq_length)

        return example['input_ids'], example['token_type_ids'], label

    else:  #if dataset == 'clue':
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
        return example['input_ids'], example['token_type_ids'], label


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default='afqmc',
        type=str,
        help="The name of the task to perform predict, selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()), )
    parser.add_argument(
        "--dataset",
        default='clue',
        type=str,
        help="The dataset of model.", )
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
        action='store_true',
        help="Whether to test performance.", )
    parser.add_argument(
        "--int8",
        action='store_true',
        help="Whether to use int8 inference.", )
    parser.add_argument(
        "--fp16",
        action='store_true',
        help="Whether to use float16 inference.", )
    args = parser.parse_args()
    return args


class Predictor(object):
    def __init__(self, predictor, input_handles, output_handles):
        self.predictor = predictor
        self.input_handles = input_handles
        self.output_handles = output_handles

    @classmethod
    def create_predictor(cls, args):
        config = paddle.inference.Config(args.model_path + "infer.pdmodel",
                                         args.model_path + "infer.pdiparams")
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

            dynamic_shape_file = os.path.join(args.model_path,
                                              'dynamic_shape.txt')
            if os.path.exists(dynamic_shape_file):
                config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file,
                                                           True)
                print('trt set dynamic shape done!')
            else:
                config.collect_shape_range_info(dynamic_shape_file)
                print(
                    'Start collect dynamic shape... Please eval again to get real result in TensorRT'
                )
                sys.exit()

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
                args.dataset,
                tokenizer,
                label_list,
                max_seq_length=args.max_seq_length)
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
                input_ids, segment_ids, label = batchify_fn(examples)
                output = self.predict_batch([input_ids, segment_ids])
                if i > args.perf_warmup_steps:
                    break
            start_time = time.time()
            for i, batch in enumerate(batches):
                examples = self.convert_predict_batch(
                    args, batch, tokenizer, batchify_fn, dataset.label_list)
                input_ids, segment_ids, _ = batchify_fn(examples)
                output = self.predict_batch([input_ids, segment_ids])

            end_time = time.time()
            sequences_num = i * args.batch_size
            print("task name: %s, time: %s qps/s, " %
                  (args.task_name, sequences_num / (end_time - start_time)))

        else:
            metric = METRIC_CLASSES[args.task_name]()
            metric.reset()
            for i, batch in enumerate(batches):
                examples = self.convert_predict_batch(
                    args, batch, tokenizer, batchify_fn, dataset.label_list)
                input_ids, segment_ids, label = batchify_fn(examples)
                output = self.predict_batch([input_ids, segment_ids])
                correct = metric.compute(
                    paddle.to_tensor(output),
                    paddle.to_tensor(np.array(label).flatten()))
                metric.update(correct)

            res = metric.accumulate()
            print("task name: %s, acc: %s, \n" % (args.task_name, res), end='')


def main():
    paddle.seed(42)
    args = parse_args()
    args.task_name = args.task_name.lower()

    predictor = Predictor.create_predictor(args)

    dev_ds = load_dataset('clue', args.task_name, splits='dev')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        Stack(dtype="int64" if dev_ds.label_list else "float32")  # label
    ): fn(samples)

    outputs = predictor.predict(dev_ds, tokenizer, batchify_fn, args)


if __name__ == "__main__":
    main()
