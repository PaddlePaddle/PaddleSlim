import os
import sys
import logging
import paddle
import argparse
import functools
import math
import time
import numpy as np

sys.path[0] = os.path.join(
    os.path.dirname("__file__"), os.path.pardir, os.path.pardir)
from paddleslim.common import get_logger
from paddleslim.quant import quant_post
from utility import add_arguments, print_arguments
import imagenet_reader as reader
_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  16,                 "Minibatch size.")
add_arg('batch_num',       int,  10,               "Batch number")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('model_path',       str,  "./inference_model/MobileNet/",  "model dir")
add_arg('save_path',       str,  "./quant_model/MobileNet/",  "model dir to save quanted model")
add_arg('model_filename',       str, None,                 "model file name")
add_arg('params_filename',      str, None,                 "params file name")
# yapf: enable


def quantize(args):
    val_reader = reader.train()

    place = paddle.CUDAPlace(0) if args.use_gpu else paddle.CPUPlace()

    assert os.path.exists(args.model_path), "args.model_path doesn't exist"
    assert os.path.isdir(args.model_path), "args.model_path must be a dir"

    exe = paddle.static.Executor(place)
    quant_post(
        executor=exe,
        model_dir=args.model_path,
        quantize_model_path=args.save_path,
        sample_generator=val_reader,
        model_filename=args.model_filename,
        params_filename=args.params_filename,
        batch_size=args.batch_size,
        batch_nums=args.batch_num)


def main():
    args = parser.parse_args()
    print_arguments(args)
    quantize(args)


if __name__ == '__main__':
    paddle.enable_static()
    main()
