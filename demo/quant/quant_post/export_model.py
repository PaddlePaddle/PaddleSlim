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
import models
from utility import add_arguments, print_arguments

_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('model',            str,  "MobileNet",                "The target model.")
add_arg('pretrained_model', str,  "../pretrained_model/MobileNetV1_pretained",                "Whether to use pretrained model.")
add_arg('data',             str, "mnist",                 "Which data to use. 'mnist' or 'imagenet'")
add_arg('test_period',      int, 10,                 "Test period in epoches.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def export_model(args):
    if args.data == "mnist":
        import paddle.dataset.mnist as reader
        train_reader = reader.train()
        val_reader = reader.test()
        class_dim = 10
        image_shape = "1,28,28"
    elif args.data == "imagenet":
        import imagenet_reader as reader
        train_reader = reader.train()
        val_reader = reader.val()
        class_dim = 1000
        image_shape = "3,224,224"
    else:
        raise ValueError("{} is not supported.".format(args.data))

    image_shape = [int(m) for m in image_shape.split(",")]
    image = paddle.static.data(
        name='image', shape=[None] + image_shape, dtype='float32')
    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    # model definition
    model = models.__dict__[args.model]()
    out = model.net(input=image, class_dim=class_dim)
    val_program = paddle.static.default_main_program().clone(for_test=True)
    place = paddle.CUDAPlace(0) if args.use_gpu else paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())

    if args.pretrained_model:
        paddle.static.load(val_program, args.pretrained_model, exe)
    else:
        assert False, "args.pretrained_model must set"

    paddle.fluid.io.save_inference_model(
        './inference_model/' + args.model,
        feeded_var_names=[image.name],
        target_vars=[out],
        executor=exe,
        main_program=val_program,
        model_filename='model',
        params_filename='weights')


def main():
    args = parser.parse_args()
    print_arguments(args)
    export_model(args)


if __name__ == '__main__':
    paddle.enable_static()
    main()
