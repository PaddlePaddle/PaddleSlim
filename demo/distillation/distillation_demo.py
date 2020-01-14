from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import logging
import paddle
import argparse
import functools
import numpy as np
import paddle.fluid as fluid
sys.path.append(sys.path[0] + "/../")
import models
from utility import add_arguments, print_arguments, _download, _decompress
from paddleslim.dist import merge, l2_loss, soft_label_loss, fsp_loss

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  64*4,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('total_images',     int,  1281167,              "Training image number.")
add_arg('image_shape',      str,  "3,224,224",         "Input image size")
add_arg('lr',               float,  0.1,               "The learning rate used to fine-tune pruned model.")
add_arg('lr_strategy',      str,  "piecewise_decay",   "The learning rate decay strategy.")
add_arg('l2_decay',         float,  3e-5,               "The l2_decay parameter.")
add_arg('momentum_rate',    float,  0.9,               "The value of momentum_rate.")
add_arg('num_epochs',       int,  120,               "The number of total epochs.")
add_arg('data',             str, "cifar10",                 "Which data to use. 'cifar10' or 'imagenet'")
add_arg('log_period',       int, 20,                 "Log period in batches.")
add_arg('model',            str,  "MobileNet",          "Set the network to use.")
add_arg('pretrained_model', str,  None,                "Whether to use pretrained model.")
add_arg('teacher_model',    str,  "ResNet50",          "Set the teacher network to use.")
add_arg('teacher_pretrained_model', str,  "./ResNet50_pretrained",                "Whether to use pretrained model.")
parser.add_argument('--step_epochs', nargs='+', type=int, default=[30, 60, 90], help="piecewise decay step")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


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
    if args.data == "cifar10":
        import paddle.dataset.cifar as reader
        train_reader = reader.train10()
        val_reader = reader.test10()
        class_dim = 10
        image_shape = "3,32,32"
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
    student_program = fluid.Program()
    s_startup = fluid.Program()

    with fluid.program_guard(student_program, s_startup):
        with fluid.unique_name.guard():
            image = fluid.layers.data(
                name='image', shape=image_shape, dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            train_loader = fluid.io.DataLoader.from_generator(
                feed_list=[image, label],
                capacity=64,
                use_double_buffer=True,
                iterable=True)
            valid_loader = fluid.io.DataLoader.from_generator(
                feed_list=[image, label],
                capacity=64,
                use_double_buffer=True,
                iterable=True)
            # model definition
            model = models.__dict__[args.model]()
            out = model.net(input=image, class_dim=class_dim)
            cost = fluid.layers.cross_entropy(input=out, label=label)
            avg_cost = fluid.layers.mean(x=cost)
            acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
            acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
    #print("="*50+"student_model_params"+"="*50)
    #for v in student_program.list_vars():
    #    print(v.name, v.shape)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    train_reader = paddle.batch(
        train_reader, batch_size=args.batch_size, drop_last=True)
    val_reader = paddle.batch(
        val_reader, batch_size=args.batch_size, drop_last=True)
    val_program = student_program.clone(for_test=True)

    places = fluid.cuda_places() if args.use_gpu else fluid.cpu_places()
    train_loader.set_sample_list_generator(train_reader, places)
    valid_loader.set_sample_list_generator(val_reader, place)

    teacher_model = models.__dict__[args.teacher_model]()
    # define teacher program
    teacher_program = fluid.Program()
    t_startup = fluid.Program()
    with fluid.program_guard(teacher_program, t_startup):
        with fluid.unique_name.guard():
            image = fluid.layers.data(
                name='image', shape=image_shape, dtype='float32')
            predict = teacher_model.net(image, class_dim=class_dim)

    #print("="*50+"teacher_model_params"+"="*50)
    #for v in teacher_program.list_vars():
    #    print(v.name, v.shape)

    exe.run(t_startup)
    _download(
        'http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar',
        '.')
    _decompress('./ResNet50_pretrained.tar')
    assert args.teacher_pretrained_model and os.path.exists(
        args.teacher_pretrained_model
    ), "teacher_pretrained_model should be set when teacher_model is not None."

    def if_exist(var):
        return os.path.exists(
            os.path.join(args.teacher_pretrained_model, var.name)
        ) and var.name != 'fc_0.w_0' and var.name != 'fc_0.b_0'

    fluid.io.load_vars(
        exe,
        args.teacher_pretrained_model,
        main_program=teacher_program,
        predicate=if_exist)

    data_name_map = {'image': 'image'}
    merge(teacher_program, student_program, data_name_map, place)

    with fluid.program_guard(student_program, s_startup):
        l2_loss = l2_loss("teacher_fc_0.tmp_0", "fc_0.tmp_0", student_program)
        loss = avg_cost + l2_loss
        opt = create_optimizer(args)
        opt.minimize(loss)
    exe.run(s_startup)
    build_strategy = fluid.BuildStrategy()
    build_strategy.fuse_all_reduce_ops = False
    parallel_main = fluid.CompiledProgram(student_program).with_data_parallel(
        loss_name=loss.name, build_strategy=build_strategy)

    for epoch_id in range(args.num_epochs):
        for step_id, data in enumerate(train_loader):
            loss_1, loss_2, loss_3 = exe.run(
                parallel_main,
                feed=data,
                fetch_list=[loss.name, avg_cost.name, l2_loss.name])
            if step_id % args.log_period == 0:
                _logger.info(
                    "train_epoch {} step {} loss {:.6f}, class loss {:.6f}, l2 loss {:.6f}".
                    format(epoch_id, step_id, loss_1[0], loss_2[0], loss_3[0]))
        val_acc1s = []
        val_acc5s = []
        for step_id, data in enumerate(valid_loader):
            val_loss, val_acc1, val_acc5 = exe.run(
                val_program,
                data,
                fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])
            val_acc1s.append(val_acc1)
            val_acc5s.append(val_acc5)
            if step_id % args.log_period == 0:
                _logger.info(
                    "valid_epoch {} step {} loss {:.6f}, top1 {:.6f}, top5 {:.6f}".
                    format(epoch_id, step_id, val_loss[0], val_acc1[0],
                           val_acc5[0]))
        _logger.info("epoch {} top1 {:.6f}, top5 {:.6f}".format(
            epoch_id, np.mean(val_acc1s), np.mean(val_acc5s)))


def main():
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()
