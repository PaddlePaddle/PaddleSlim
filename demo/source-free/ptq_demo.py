import os
import sys
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)
import argparse
import functools
from functools import partial

import numpy as np
from collections import namedtuple, Iterable
import paddle
import paddle.nn as nn
from paddle.io import Dataset, BatchSampler, DataLoader
import imagenet_reader as reader
from paddleslim.source_free.auto_compression import AutoCompression
from paddleslim.source_free.strategy_config import QuantizationConfig, HyperParameterOptimizationConfig, TrainConfig
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('model_dir',                   str,    None,         "inference model directory.")
add_arg('model_filename',              str,    None,         "inference model filename.")
add_arg('params_filename',             str,    None,         "inference params filename.")
add_arg('save_dir',                    str,    None,         "directory to save compressed model.")
add_arg('devices',                     str,    'gpu',        "which device used to compress.")
add_arg('batch_size',                  int,    None,         "batch size in dataloader.")
add_arg('ptq_algo',                    str,    None,         "algorithm of post training quantization.", nargs="+")
add_arg('bias_correct',                bool,   None,         "whether to use bias correct.", nargs="+")
add_arg('weight_quantize_type',        str,    None,         "weight quantization type.", nargs="+")
add_arg('hist_percent',                float,  None,         "mininum histogram percent and max histogram percent.", nargs="+")
add_arg('batch_num',                   int,    None,         "mininum batch number and max batch number.", nargs="+")
add_arg('max_quant_count',             int,    None,         "max quantization count.")
# yapf: enable


def reader_wrapper(reader):
    def gen():
        for i, data in enumerate(reader()):
            imgs = np.float32([item[0] for item in data])
            yield {"inputs": imgs}

    return gen


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    paddle.enable_static()

    default_ptq_config = {
        "quantize_op_types": ["conv2d", "depthwise_conv2d", "mul"],
        "weight_bits": 8,
        "activation_bits": 8,
    }

    default_hpo_config = {
        "ptq_algo": args.ptq_algo,
        "bias_correct": args.bias_correct,
        "weight_quantize_type": args.weight_quantize_type,
        "hist_percent": args.hist_percent,
        "batch_num": args.batch_num,
        "batch_size": [args.batch_size],
        "max_quant_count": args.max_quant_count
    }
    train_reader = paddle.batch(reader.train(), batch_size=args.batch_size)
    eval_reader = paddle.batch(reader.val(), batch_size=args.batch_size)
    train_dataloader = reader_wrapper(train_reader)
    eval_dataloader = reader_wrapper(eval_reader)

    ac = AutoCompression(
        model_dir=args.model_dir,
        model_filename=args.model_filename,
        params_filename=args.params_filename,
        save_dir=args.save_dir,
        strategy_config={
            "QuantizationConfig": QuantizationConfig(**default_ptq_config),
            "HyperParameterOptimizationConfig":
            HyperParameterOptimizationConfig(**default_hpo_config)
        },
        train_config=None,
        train_dataloader=train_dataloader,
        eval_callback=eval_dataloader,
        devices=args.devices)

    ac.compression()
