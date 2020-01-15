import os
import sys
import logging
import paddle
import argparse
import functools
import math
import time
import numpy as np
import paddle.fluid as fluid
from paddleslim.prune import load_model
from paddleslim.common import get_logger
from paddleslim.analysis import flops
sys.path.append(sys.path[0] + "/../")
import models
from utility import add_arguments, print_arguments

_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  64 * 4,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('model',            str,  "MobileNet",                "The target model.")
add_arg('model_path', str,  "./models/0",                "The path of model used to evalate..")
add_arg('data',             str, "mnist",                 "Which data to use. 'mnist' or 'imagenet'")
add_arg('log_period',       int, 10,                 "Log period in batches.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def eval(args):
    train_reader = None
    test_reader = None
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
    assert args.model in model_list, "{} is not in lists: {}".format(
        args.model, model_list)
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    # model definition
    model = models.__dict__[args.model]()
    out = model.net(input=image, class_dim=class_dim)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
    val_program = fluid.default_main_program().clone(for_test=True)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    val_reader = paddle.batch(val_reader, batch_size=args.batch_size)

    val_feeder = feeder = fluid.DataFeeder(
        [image, label], place, program=val_program)

    load_model(val_program, "./model/mobilenetv1_prune_50")

    batch_id = 0
    acc_top1_ns = []
    acc_top5_ns = []
    for data in val_reader():
        start_time = time.time()
        acc_top1_n, acc_top5_n = exe.run(
            val_program,
            feed=val_feeder.feed(data),
            fetch_list=[acc_top1.name, acc_top5.name])
        end_time = time.time()
        if batch_id % args.log_period == 0:
            _logger.info(
                "Eval batch[{}] - acc_top1: {}; acc_top5: {}; time: {}".format(
                    batch_id,
                    np.mean(acc_top1_n),
                    np.mean(acc_top5_n), end_time - start_time))
        acc_top1_ns.append(np.mean(acc_top1_n))
        acc_top5_ns.append(np.mean(acc_top5_n))
        batch_id += 1

    _logger.info("Final eval - acc_top1: {}; acc_top5: {}".format(
        np.mean(np.array(acc_top1_ns)), np.mean(np.array(acc_top5_ns))))


def main():
    args = parser.parse_args()
    print_arguments(args)
    eval(args)


if __name__ == '__main__':
    main()
