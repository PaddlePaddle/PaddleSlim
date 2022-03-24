import os
import sys
import math
import time
import numpy as np
import paddle
import logging
import argparse
import functools

sys.path[0] = os.path.join(
    os.path.dirname("__file__"), os.path.pardir, os.path.pardir)
sys.path[1] = os.path.join(
    os.path.dirname("__file__"), os.path.pardir, os.path.pardir, os.path.pardir)
from paddleslim.common import get_logger
from paddleslim.quant import quant_post_hpo
from utility import add_arguments, print_arguments
import imagenet_reader as reader
_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('model_path',       str,  "./inference_model/MobileNet/",  "model dir")
add_arg('save_path',       str,  "./quant_model/MobileNet/",  "model dir to save quanted model")
add_arg('model_filename',       str, None,                 "model file name")
add_arg('params_filename',      str, None,                 "params file name")
add_arg('max_model_quant_count',    int, 30,                 "max model quant count")

def quantize(args):
    place = paddle.CUDAPlace(0) if args.use_gpu else paddle.CPUPlace()

    assert os.path.exists(args.model_path), "args.model_path doesn't exist"
    assert os.path.isdir(args.model_path), "args.model_path must be a dir"

    def reader_generator(imagenet_reader):
        def gen():
            for i, data in enumerate(imagenet_reader()):
                image, label = data
                image = np.expand_dims(image, axis=0)
                yield image
        return gen

    exe = paddle.static.Executor(place)
    quant_post_hpo(
        exe,
        place,
        args.model_path,
        args.save_path,
        train_sample_generator=reader_generator(reader.train()),
        eval_sample_generator=reader_generator(reader.val()),
        model_filename=args.model_filename,
        params_filename=args.params_filename,
        save_model_filename='__model__',
        save_params_filename='__params__',
        quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
        weight_quantize_type=['channel_wise_abs_max'],
        runcount_limit=args.max_model_quant_count)

def main():
    args = parser.parse_args()
    print_arguments(args)
    quantize(args)


if __name__ == '__main__':
    paddle.enable_static()
    main()
