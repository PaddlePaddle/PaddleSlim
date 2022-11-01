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


def parse_args():
    """
    parse_args func
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="./afqmc",
        type=str,
        required=True,
        help="The path prefix of inference model to be used.", )
    parser.add_argument(
        "--model_filename",
        type=str,
        default="inference.pdmodel",
        help="model file name")
    parser.add_argument(
        "--params_filename",
        type=str,
        default="inference.pdiparams",
        help="params file name")
    parser.add_argument(
        "--task_name",
        default="afqmc",
        type=str,
        help="The name of the task to perform predict, selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()), )
    parser.add_argument(
        "--dataset",
        default="clue",
        type=str,
        help="The dataset of model.", )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu"],
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
        action="store_true",
        help="Whether to use inference engin TensorRT.", )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "int8"],
        help="The precision of inference. It can be 'fp32', 'fp16' or 'int8'. Default is 'fp16'.",
    )
    parser.add_argument(
        "--use_mkldnn",
        type=bool,
        default=False,
        help="Whether use mkldnn or not.")
    parser.add_argument(
        "--cpu_threads", type=int, default=1, help="Num of cpu threads.")
    args = parser.parse_args()
    return args


def _convert_example(example,
                     dataset,
                     tokenizer,
                     label_list,
                     max_seq_length=512):
    assert dataset in ["glue", "clue"
                       ], "This demo only supports for dataset glue or clue"
    """Convert a glue example into necessary features."""
    if dataset == "glue":
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example["labels"]
        label = np.array([label], dtype=label_dtype)
        # Convert raw text to feature
        example = tokenizer(example["sentence"], max_seq_len=max_seq_length)

        return example["input_ids"], example["token_type_ids"], label

    else:  # if dataset == 'clue':
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        example["label"] = np.array(
            example["label"], dtype="int64").reshape((-1, 1))
        label = example["label"]
        # Convert raw text to feature
        if "keyword" in example:  # CSL
            sentence1 = " ".join(example["keyword"])
            example = {
                "sentence1": sentence1,
                "sentence2": example["abst"],
                "label": example["label"]
            }
        elif "target" in example:  # wsc
            text, query, pronoun, query_idx, pronoun_idx = (
                example["text"],
                example["target"]["span1_text"],
                example["target"]["span2_text"],
                example["target"]["span1_index"],
                example["target"]["span2_index"], )
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
            example["sentence"] = text
        if tokenizer is None:
            return example
        if "sentence" in example:
            example = tokenizer(example["sentence"], max_seq_len=max_seq_length)
        elif "sentence1" in example:
            example = tokenizer(
                example["sentence1"],
                text_pair=example["sentence2"],
                max_seq_len=max_seq_length)
        return example["input_ids"], example["token_type_ids"], label


class Predictor(object):
    """
    Inference Predictor class
    """

    def __init__(self, predictor, input_handles, output_handles):
        self.predictor = predictor
        self.input_handles = input_handles
        self.output_handles = output_handles

    @classmethod
    def create_predictor(cls, args):
        """
        create_predictor func
        """
        cls.rerun_flag = False
        config = paddle.inference.Config(
            os.path.join(args.model_path, args.model_filename),
            os.path.join(args.model_path, args.params_filename))
        if args.device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
            cls.device = paddle.set_device("gpu")
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(args.cpu_threads)
            config.switch_ir_optim()
            if args.use_mkldnn:
                config.enable_mkldnn()
                if args.precision == "int8":
                    config.enable_mkldnn_int8()

        precision_map = {
            "int8": inference.PrecisionType.Int8,
            "fp32": inference.PrecisionType.Float32,
            "fp16": inference.PrecisionType.Half,
        }
        if args.precision in precision_map.keys() and args.use_trt:
            config.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=args.batch_size,
                min_subgraph_size=5,
                precision_mode=precision_map[args.precision],
                use_static=True,
                use_calib_mode=False, )

            dynamic_shape_file = os.path.join(args.model_path,
                                              "dynamic_shape.txt")
            if os.path.exists(dynamic_shape_file):
                config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file,
                                                           True)
                print("trt set dynamic shape done!")
            else:
                config.collect_shape_range_info(dynamic_shape_file)
                print("Start collect dynamic shape...")
                cls.rerun_flag = True

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
        """
        predict from batch func
        """
        for input_field, input_handle in zip(data, self.input_handles):
            input_handle.copy_from_cpu(input_field)
        self.predictor.run()
        output = [
            output_handle.copy_to_cpu() for output_handle in self.output_handles
        ]
        return output

    def _convert_predict_batch(self, args, data, tokenizer, batchify_fn,
                               label_list):
        examples = []
        for example in data:
            example = _convert_example(
                example,
                args.dataset,
                tokenizer,
                label_list,
                max_seq_length=args.max_seq_length)
            examples.append(example)

        return examples

    def predict(self, dataset, tokenizer, batchify_fn, args):
        """
        predict func
        """
        batches = [
            dataset[idx:idx + args.batch_size]
            for idx in range(0, len(dataset), args.batch_size)
        ]

        for i, batch in enumerate(batches):
            examples = self._convert_predict_batch(
                args, batch, tokenizer, batchify_fn, dataset.label_list)
            input_ids, segment_ids, label = batchify_fn(examples)
            output = self.predict_batch([input_ids, segment_ids])
            if i > args.perf_warmup_steps:
                break
            if self.rerun_flag:
                return

        metric = METRIC_CLASSES[args.task_name]()
        metric.reset()
        predict_time = 0.0
        for i, batch in enumerate(batches):
            examples = self._convert_predict_batch(
                args, batch, tokenizer, batchify_fn, dataset.label_list)
            input_ids, segment_ids, label = batchify_fn(examples)
            start_time = time.time()
            output = self.predict_batch([input_ids, segment_ids])
            end_time = time.time()
            predict_time += end_time - start_time
            correct = metric.compute(
                paddle.to_tensor(output),
                paddle.to_tensor(np.array(label).flatten()))
            metric.update(correct)

        sequences_num = i * args.batch_size
        print(
            "[benchmark]task name: {}, batch size: {} Inference time per batch: {}ms, qps: {}.".
            format(
                args.task_name,
                args.batch_size,
                round(predict_time * 1000 / i, 2),
                round(sequences_num / predict_time, 2), ))
        res = metric.accumulate()
        print(
            "[benchmark]task name: %s, acc: %s. \n" % (args.task_name, res),
            end="")
        sys.stdout.flush()


def main():
    """
    main func
    """
    paddle.seed(42)
    args = parse_args()
    args.task_name = args.task_name.lower()
    if args.use_mkldnn:
        paddle.set_device("cpu")

    predictor = Predictor.create_predictor(args)

    dev_ds = load_dataset("clue", args.task_name, splits="dev")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        Stack(dtype="int64" if dev_ds.label_list else "float32"),  # label
    ): fn(samples)

    predictor.predict(dev_ds, tokenizer, batchify_fn, args)
    if predictor.rerun_flag:
        print(
            "***** Collect dynamic shape done, Please rerun the program to get correct results. *****"
        )


if __name__ == "__main__":
    paddle.set_device("cpu")
    main()
