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
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
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
    """
    parse_args func
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="./x2paddle_cola",
        type=str,
        required=True,
        help="The path prefix of inference model to be used.", )
    parser.add_argument(
        "--model_filename",
        type=str,
        default="model.pdmodel",
        help="model file name")
    parser.add_argument(
        "--params_filename",
        type=str,
        default="model.pdiparams",
        help="params file name")
    parser.add_argument(
        "--task_name",
        default="cola",
        type=str,
        help="The name of the task to perform predict, selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()), )
    parser.add_argument(
        "--model_type",
        default="bert-base-cased",
        type=str,
        help="Model type selected in bert.")
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-cased",
        type=str,
        help="The directory or name of model.", )
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


def _convert_example(
        example,
        tokenizer,
        label_list,
        max_seq_length=512,
        task_name=None,
        is_test=False,
        padding="max_length",
        return_attention_mask=True, ):
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example["labels"]
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    sentence1_key, sentence2_key = task_to_keys[task_name]
    texts = (example[sentence1_key], ) if sentence2_key is None else (
        example[sentence1_key], example[sentence2_key])
    example = tokenizer(
        *texts,
        max_seq_len=max_seq_length,
        padding=padding,
        return_attention_mask=return_attention_mask)
    if not is_test:
        if return_attention_mask:
            return example["input_ids"], example["attention_mask"], example[
                "token_type_ids"], label
        else:
            return example["input_ids"], example["token_type_ids"], label
    else:
        if return_attention_mask:
            return example["input_ids"], example["attention_mask"], example[
                "token_type_ids"]
        else:
            return example["input_ids"], example["token_type_ids"]


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
                    config.enable_mkldnn_int8(
                        {"fc", "reshape2", "transpose2", "slice"})

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

    def predict(self, dataset, collate_fn, args):
        """
        predict func
        """
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=args.batch_size, shuffle=False)
        data_loader = paddle.io.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=0,
            return_list=True)

        for i, data in enumerate(data_loader):
            for input_field, input_handle in zip(data, self.input_handles):
                input_handle.copy_from_cpu(input_field.numpy() if isinstance(
                    input_field, paddle.Tensor) else input_field)
            self.predictor.run()
            output = [
                output_handle.copy_to_cpu()
                for output_handle in self.output_handles
            ]
            if i > args.perf_warmup_steps:
                break
            if self.rerun_flag:
                return

        metric = METRIC_CLASSES[args.task_name]()
        metric.reset()
        predict_time = 0.0
        for i, data in enumerate(data_loader):
            for input_field, input_handle in zip(data, self.input_handles):
                input_handle.copy_from_cpu(input_field.numpy() if isinstance(
                    input_field, paddle.Tensor) else input_field)
            start_time = time.time()
            self.predictor.run()
            output = [
                output_handle.copy_to_cpu()
                for output_handle in self.output_handles
            ]
            end_time = time.time()
            predict_time += end_time - start_time
            label = data[-1]
            correct = metric.compute(
                paddle.to_tensor(output[0]),
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
    if args.use_mkldnn:
        paddle.set_device("cpu")

    predictor = Predictor.create_predictor(args)

    args.task_name = args.task_name.lower()
    args.model_type = args.model_type.lower()

    dev_ds = load_dataset("glue", args.task_name, splits="dev")
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    trans_func = partial(
        _convert_example,
        tokenizer=tokenizer,
        label_list=dev_ds.label_list,
        max_seq_length=args.max_seq_length,
        task_name=args.task_name,
        return_attention_mask=True, )

    dev_ds = dev_ds.map(trans_func)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=0),
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        Stack(dtype="int64" if dev_ds.label_list else "float32"),  # label
    ): fn(samples)
    predictor.predict(dev_ds, batchify_fn, args)


if __name__ == "__main__":
    paddle.set_device("cpu")
    main()
