import os
import sys
import logging
import paddle
import argparse
import functools
import math
import paddle.fluid as fluid
import imagenet_reader as reader
import models
from utility import add_arguments, print_arguments
import numpy as np
import time
from paddleslim.prune import Pruner
from paddleslim.analysis import flops

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  64 * 4,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('model',            str,  "MobileNet",                "The target model.")
add_arg('model_save_dir',            str,  "./",                "checkpoint  model.")
add_arg('pretrained_model', str,  "../pretrained_model/MobileNetV1_pretained",                "Whether to use pretrained model.")
add_arg('lr',               float,  0.01,               "The learning rate used to fine-tune pruned model.")
add_arg('lr_strategy',      str,  "piecewise_decay",   "The learning rate decay strategy.")
add_arg('l2_decay',         float,  3e-5,               "The l2_decay parameter.")
add_arg('momentum_rate',    float,  0.9,               "The value of momentum_rate.")
add_arg('num_epochs',       int,  20,               "The number of total epochs.")
add_arg('total_images',     int,  1281167,               "The number of total training images.")
parser.add_argument('--step_epochs', nargs='+', type=int, default=[5, 15], help="piecewise decay step")
add_arg('config_file',      str, None,                 "The config file for compression with yaml format.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]
ratiolist = [
    #    [0.06, 0.0, 0.09, 0.03, 0.09, 0.02, 0.05, 0.03, 0.0, 0.07, 0.07, 0.05, 0.08],
    #    [0.08, 0.02, 0.03, 0.13, 0.1, 0.06, 0.03, 0.04, 0.14, 0.02, 0.03, 0.02, 0.01],
]


def save_model(args, exe, train_prog, eval_prog, info):
    model_path = os.path.join(args.model_save_dir, args.model, str(info))
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    fluid.io.save_persistables(exe, model_path, main_program=train_prog)
    print("Already save model in %s" % (model_path))


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
    class_dim = 1000
    image_shape = "3,224,224"
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
            exist = os.path.exists(
                os.path.join(args.pretrained_model, var.name))
            print("exist", exist)
            return exist

        #fluid.io.load_vars(exe, args.pretrained_model, predicate=if_exist)

    val_reader = paddle.batch(reader.val(), batch_size=args.batch_size)
    train_reader = paddle.batch(
        reader.train(), batch_size=args.batch_size, drop_last=True)

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
            print(
                "Eval epoch[{}] batch[{}] - acc_top1: {}; acc_top5: {}; time: {}".
                format(epoch, batch_id,
                       np.mean(acc_top1_n),
                       np.mean(acc_top5_n), end_time - start_time))
            acc_top1_ns.append(np.mean(acc_top1_n))
            acc_top5_ns.append(np.mean(acc_top5_n))
            batch_id += 1

        print("Final eval epoch[{}] - acc_top1: {}; acc_top5: {}".format(
            epoch,
            np.mean(np.array(acc_top1_ns)), np.mean(np.array(acc_top5_ns))))

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
            loss_n, acc_top1_n, acc_top5_n, lr_n = exe.run(
                train_program,
                feed=train_feeder.feed(data),
                fetch_list=[
                    avg_cost.name, acc_top1.name, acc_top5.name,
                    "learning_rate"
                ])
            end_time = time.time()
            loss_n = np.mean(loss_n)
            acc_top1_n = np.mean(acc_top1_n)
            acc_top5_n = np.mean(acc_top5_n)
            lr_n = np.mean(lr_n)
            print(
                "epoch[{}]-batch[{}] - loss: {}; acc_top1: {}; acc_top5: {};lrn: {}; time: {}".
                format(epoch, batch_id, loss_n, acc_top1_n, acc_top5_n, lr_n,
                       end_time - start_time))
            batch_id += 1

    params = []
    for param in fluid.default_main_program().global_block().all_parameters():
        #if "_weights" in  param.name and "conv1_weights" not in param.name:
        if "_sep_weights" in param.name:
            params.append(param.name)
    print("fops before pruning: {}".format(
        flops(fluid.default_main_program())))
    pruned_program_iter = fluid.default_main_program()
    pruned_val_program_iter = val_program
    for ratios in ratiolist:
        pruner = Pruner()
        pruned_val_program_iter = pruner.prune(
            pruned_val_program_iter,
            fluid.global_scope(),
            params=params,
            ratios=ratios,
            place=place,
            only_graph=True)

        pruned_program_iter = pruner.prune(
            pruned_program_iter,
            fluid.global_scope(),
            params=params,
            ratios=ratios,
            place=place)
        print("fops after pruning: {}".format(flops(pruned_program_iter)))
    """ do not inherit learning rate """
    if (os.path.exists(args.pretrained_model + "/learning_rate")):
        os.remove(args.pretrained_model + "/learning_rate")
    if (os.path.exists(args.pretrained_model + "/@LR_DECAY_COUNTER@")):
        os.remove(args.pretrained_model + "/@LR_DECAY_COUNTER@")
    fluid.io.load_vars(
        exe,
        args.pretrained_model,
        main_program=pruned_program_iter,
        predicate=if_exist)

    pruned_program = pruned_program_iter
    pruned_val_program = pruned_val_program_iter
    for i in range(args.num_epochs):
        train(i, pruned_program)
        test(i, pruned_val_program)
        save_model(args, exe, pruned_program, pruned_val_program, i)


def main():
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()
