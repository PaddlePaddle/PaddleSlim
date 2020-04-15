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
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)
from paddleslim.prune import Pruner, save_model
from paddleslim.common import get_logger
from paddleslim.analysis import flops
import models
from utility import add_arguments, print_arguments

_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  64 * 4,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('model',            str,  "MobileNet",                "The target model.")
add_arg('pretrained_model', str,  "../pretrained_model/MobileNetV1_pretained",                "Whether to use pretrained model.")
add_arg('lr',               float,  0.1,               "The learning rate used to fine-tune pruned model.")
add_arg('lr_strategy',      str,  "piecewise_decay",   "The learning rate decay strategy.")
add_arg('l2_decay',         float,  3e-5,               "The l2_decay parameter.")
add_arg('momentum_rate',    float,  0.9,               "The value of momentum_rate.")
add_arg('num_epochs',       int,  120,               "The number of total epochs.")
add_arg('total_images',     int,  1281167,               "The number of total training images.")
parser.add_argument('--step_epochs', nargs='+', type=int, default=[30, 60, 90], help="piecewise decay step")
add_arg('config_file',      str, None,                 "The config file for compression with yaml format.")
add_arg('data',             str, "mnist",                 "Which data to use. 'mnist' or 'imagenet'")
add_arg('log_period',       int, 10,                 "Log period in batches.")
add_arg('test_period',      int, 10,                 "Test period in epoches.")
add_arg('model_path',       str, "./models",         "The path to save model.")
add_arg('pruned_ratio',     float, None,         "The ratios to be pruned.")
add_arg('criterion',        str, "l1_norm",         "The prune criterion to be used, support l1_norm and batch_norm_scale.")
add_arg('save_inference',   bool, False,                "Whether to save inference model.")
# yapf: enable

model_list = models.__all__


def get_pruned_params(args, program):
    params = []
    if args.model == "MobileNet":
        for param in program.global_block().all_parameters():
            if "_sep_weights" in param.name:
                params.append(param.name)
    elif args.model == "MobileNetV2":
        for param in program.global_block().all_parameters():
            if "linear_weights" in param.name or "expand_weights" in param.name:
                params.append(param.name)
    elif args.model == "ResNet34":
        for param in program.global_block().all_parameters():
            if "weights" in param.name and "branch" in param.name:
                params.append(param.name)
    elif args.model == "PVANet":
        for param in program.global_block().all_parameters():
            if "conv_weights" in param.name:
                params.append(param.name)
    return params


def piecewise_decay(args):
    step = int(math.ceil(float(args.total_images) / args.batch_size))
    bd = [step * e for e in args.step_epochs]
    lr = [args.lr * (0.1**i) for i in range(len(bd) + 1)]
    learning_rate = fluid.layers.piecewise_decay(boundaries=bd, values=lr)

    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        regularization=fluid.regularizer.L2Decay(args.l2_decay))
    return optimizer


def cosine_decay(args):
    step = int(math.ceil(float(args.total_images) / args.batch_size))
    learning_rate = fluid.layers.cosine_decay(
        learning_rate=args.lr, step_each_epoch=step, epochs=args.num_epochs)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        regularization=fluid.regularizer.L2Decay(args.l2_decay))
    return optimizer


def create_optimizer(args):
    if args.lr_strategy == "piecewise_decay":
        return piecewise_decay(args)
    elif args.lr_strategy == "cosine_decay":
        return cosine_decay(args)


def compress(args):
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
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
    val_program = fluid.default_main_program().clone(for_test=True)
    opt = create_optimizer(args)
    opt.minimize(avg_cost)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if args.pretrained_model:

        def if_exist(var):
            return os.path.exists(
                os.path.join(args.pretrained_model, var.name))

        _logger.info("Load pretrained model from {}".format(
            args.pretrained_model))
        fluid.io.load_vars(exe, args.pretrained_model, predicate=if_exist)

    val_reader = paddle.batch(val_reader, batch_size=args.batch_size)
    train_reader = paddle.batch(
        train_reader, batch_size=args.batch_size, drop_last=True)

    train_feeder = feeder = fluid.DataFeeder([image, label], place)
    val_feeder = feeder = fluid.DataFeeder(
        [image, label], place, program=val_program)

    def test(epoch, program):
        batch_id = 0
        acc_top1_ns = []
        acc_top5_ns = []
        for data in val_reader():
            start_time = time.time()
            acc_top1_n, acc_top5_n = exe.run(
                program,
                feed=train_feeder.feed(data),
                fetch_list=[acc_top1.name, acc_top5.name])
            end_time = time.time()
            if batch_id % args.log_period == 0:
                _logger.info(
                    "Eval epoch[{}] batch[{}] - acc_top1: {}; acc_top5: {}; time: {}".
                    format(epoch, batch_id,
                           np.mean(acc_top1_n),
                           np.mean(acc_top5_n), end_time - start_time))
            acc_top1_ns.append(np.mean(acc_top1_n))
            acc_top5_ns.append(np.mean(acc_top5_n))
            batch_id += 1

        _logger.info("Final eval epoch[{}] - acc_top1: {}; acc_top5: {}".
                     format(epoch,
                            np.mean(np.array(acc_top1_ns)),
                            np.mean(np.array(acc_top5_ns))))

    def train(epoch, program):

        build_strategy = fluid.BuildStrategy()
        exec_strategy = fluid.ExecutionStrategy()
        train_program = fluid.compiler.CompiledProgram(
            program).with_data_parallel(
                loss_name=avg_cost.name,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)

        batch_id = 0
        for data in train_reader():
            start_time = time.time()
            loss_n, acc_top1_n, acc_top5_n = exe.run(
                train_program,
                feed=train_feeder.feed(data),
                fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])
            end_time = time.time()
            loss_n = np.mean(loss_n)
            acc_top1_n = np.mean(acc_top1_n)
            acc_top5_n = np.mean(acc_top5_n)
            if batch_id % args.log_period == 0:
                _logger.info(
                    "epoch[{}]-batch[{}] - loss: {}; acc_top1: {}; acc_top5: {}; time: {}".
                    format(epoch, batch_id, loss_n, acc_top1_n, acc_top5_n,
                           end_time - start_time))
            batch_id += 1

    test(0, val_program)

    params = get_pruned_params(args, fluid.default_main_program())
    _logger.info("FLOPs before pruning: {}".format(
        flops(fluid.default_main_program())))
    pruner = Pruner(args.criterion)
    pruned_val_program, _, _ = pruner.prune(
        val_program,
        fluid.global_scope(),
        params=params,
        ratios=[args.pruned_ratio] * len(params),
        place=place,
        only_graph=True)

    pruned_program, _, _ = pruner.prune(
        fluid.default_main_program(),
        fluid.global_scope(),
        params=params,
        ratios=[args.pruned_ratio] * len(params),
        place=place)
    _logger.info("FLOPs after pruning: {}".format(flops(pruned_program)))
    for i in range(args.num_epochs):
        train(i, pruned_program)
        if i % args.test_period == 0:
            test(i, pruned_val_program)
            save_model(exe, pruned_val_program,
                       os.path.join(args.model_path, str(i)))
        if args.save_inference:
            infer_model_path = os.path.join(args.model_path, "infer_models",
                                            str(i))
            fluid.io.save_inference_model(infer_model_path, ["image"], [out],
                                          exe, pruned_val_program)
            _logger.info("Saved inference model into [{}]".format(
                infer_model_path))


def main():
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()
