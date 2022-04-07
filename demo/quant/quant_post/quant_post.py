import os
import sys
import logging
import paddle
import argparse
import functools
import math
import time
import random
import numpy as np
import paddle

sys.path[0] = os.path.join(
    os.path.dirname("__file__"), os.path.pardir, os.path.pardir)
from paddleslim.common import get_logger
from paddleslim.quant import quant_post_static
from utility import add_arguments, print_arguments
import imagenet_reader as reader
_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  32,                 "Minibatch size.")
add_arg('batch_num',       int,  1,               "Batch number")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('model_path',       str,  "./inference_model/MobileNetV1_infer/",  "model dir")
add_arg('save_path',       str,  "./quant_model/MobileNet/",  "model dir to save quanted model")
add_arg('model_filename',       str, 'inference.pdmodel',                 "model file name")
add_arg('params_filename',      str, 'inference.pdiparams',                 "params file name")
add_arg('algo',         str, 'hist',               "calibration algorithm")
add_arg('round_type',         str, 'round',               "The method of converting the quantized weights.")
add_arg('hist_percent',         float, 0.9999,             "The percentile of algo:hist")
add_arg('is_full_quantize',         bool, False,             "Whether is full quantization or not.")
add_arg('bias_correction',         bool, False,             "Whether to use bias correction")
add_arg('ce_test',                 bool,   False,                                        "Whether to CE test.")
add_arg('onnx_format',             bool,   False,                  "Whether to export the quantized model with format of ONNX.")
add_arg('input_name',         str, 'inputs',               "The name of model input.")

# yapf: enable


def quantize(args):
    shuffle = True
    if args.ce_test:
        # set seed
        seed = 111
        np.random.seed(seed)
        paddle.seed(seed)
        random.seed(seed)
        shuffle = False

    place = paddle.CUDAPlace(0) if args.use_gpu else paddle.CPUPlace()
    val_dataset = reader.ImageNetDataset(mode='test')
    image_shape = [3, 224, 224]
    image = paddle.static.data(
        name=args.input_name, shape=[None] + image_shape, dtype='float32')
    data_loader = paddle.io.DataLoader(
        val_dataset,
        places=place,
        feed_list=[image],
        drop_last=False,
        return_list=False,
        batch_size=args.batch_size,
        shuffle=False)

    assert os.path.exists(args.model_path), "args.model_path doesn't exist"
    assert os.path.isdir(args.model_path), "args.model_path must be a dir"

    exe = paddle.static.Executor(place)
    quant_post_static(
        executor=exe,
        model_dir=args.model_path,
        quantize_model_path=args.save_path,
        data_loader=data_loader,
        model_filename=args.model_filename,
        params_filename=args.params_filename,
        batch_size=args.batch_size,
        batch_nums=args.batch_num,
        algo=args.algo,
        round_type=args.round_type,
        hist_percent=args.hist_percent,
        is_full_quantize=args.is_full_quantize,
        bias_correction=args.bias_correction,
        onnx_format=args.onnx_format)


def main():
    args = parser.parse_args()
    print_arguments(args)
    quantize(args)


if __name__ == '__main__':
    paddle.enable_static()
    main()
