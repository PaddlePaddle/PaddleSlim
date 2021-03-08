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
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)
import models
from utility import add_arguments, print_arguments, _download, _decompress
from paddleslim.dist import merge, l2_loss, soft_label_loss, fsp_loss

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  64,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('save_inference',   bool, False,                "Whether to save inference model.")
add_arg('total_images',     int,  1281167,              "Training image number.")
add_arg('image_shape',      str,  "3,224,224",         "Input image size")
add_arg('lr',               float,  0.1,               "The learning rate used to fine-tune pruned model.")
add_arg('lr_strategy',      str,  "piecewise_decay",   "The learning rate decay strategy.")
add_arg('l2_decay',         float,  3e-5,               "The l2_decay parameter.")
add_arg('momentum_rate',    float,  0.9,               "The value of momentum_rate.")
add_arg('num_epochs',       int,  120,               "The number of total epochs.")
add_arg('data',             str, "imagenet",                 "Which data to use. 'cifar10' or 'imagenet'")
add_arg('log_period',       int,  20,                 "Log period in batches.")
add_arg('model',            str,  "MobileNet",          "Set the network to use.")
add_arg('pretrained_model', str,  None,                "Whether to use pretrained model.")
add_arg('teacher_model',    str,  "ResNet50_vd",          "Set the teacher network to use.")
add_arg('teacher_pretrained_model', str,  "./ResNet50_vd_pretrained",                "Whether to use pretrained model.")
parser.add_argument('--step_epochs', nargs='+', type=int, default=[30, 60, 90], help="piecewise decay step")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def piecewise_decay(args):
    if args.use_gpu:
        devices_num = paddle.fluid.core.get_cuda_device_count()
    else:
        devices_num = int(os.environ.get('CPU_NUM', 1))
    step = int(
        math.ceil(float(args.total_images) / args.batch_size) / devices_num)
    bd = [step * e for e in args.step_epochs]
    lr = [args.lr * (0.1**i) for i in range(len(bd) + 1)]
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=bd, values=lr, verbose=False)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay))
    return learning_rate, optimizer


def cosine_decay(args):
    if args.use_gpu:
        devices_num = paddle.fluid.core.get_cuda_device_count()
    else:
        devices_num = int(os.environ.get('CPU_NUM', 1))
    step = int(
        math.ceil(float(args.total_images) / args.batch_size) / devices_num)
    learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.lr, T_max=step * args.num_epochs, verbose=False)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay))
    return learning_rate, optimizer


def create_optimizer(args):
    if args.lr_strategy == "piecewise_decay":
        return piecewise_decay(args)
    elif args.lr_strategy == "cosine_decay":
        return cosine_decay(args)


def compress(args):
    if args.data == "cifar10":
        train_dataset = paddle.vision.datasets.Cifar10(mode='train')
        val_dataset = paddle.vision.datasets.Cifar10(mode='test')
        class_dim = 10
        image_shape = "3,32,32"
    elif args.data == "imagenet":
        import imagenet_reader as reader
        train_dataset = reader.ImageNetDataset(mode='train')
        val_dataset = reader.ImageNetDataset(mode='val')
        class_dim = 1000
        image_shape = "3,224,224"
    else:
        raise ValueError("{} is not supported.".format(args.data))
    image_shape = [int(m) for m in image_shape.split(",")]

    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    student_program = paddle.static.Program()
    s_startup = paddle.static.Program()
    places = paddle.static.cuda_places(
    ) if args.use_gpu else paddle.static.cpu_places()
    place = places[0]

    with paddle.static.program_guard(student_program, s_startup):
        with paddle.fluid.unique_name.guard():
            image = paddle.static.data(
                name='image', shape=[None] + image_shape, dtype='float32')
            label = paddle.static.data(
                name='label', shape=[None, 1], dtype='int64')
            train_loader = paddle.io.DataLoader(
                train_dataset,
                places=places,
                feed_list=[image, label],
                drop_last=True,
                batch_size=args.batch_size,
                return_list=False,
                shuffle=True,
                use_shared_memory=True,
                num_workers=4)
            valid_loader = paddle.io.DataLoader(
                val_dataset,
                places=place,
                feed_list=[image, label],
                drop_last=False,
                return_list=False,
                use_shared_memory=True,
                batch_size=args.batch_size,
                shuffle=False)
            # model definition
            model = models.__dict__[args.model]()
            out = model.net(input=image, class_dim=class_dim)
            cost = paddle.nn.functional.loss.cross_entropy(
                input=out, label=label)
            avg_cost = paddle.mean(x=cost)
            acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
            acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

    val_program = student_program.clone(for_test=True)
    exe = paddle.static.Executor(place)

    teacher_model = models.__dict__[args.teacher_model]()
    # define teacher program
    teacher_program = paddle.static.Program()
    t_startup = paddle.static.Program()
    with paddle.static.program_guard(teacher_program, t_startup):
        with paddle.fluid.unique_name.guard():
            image = paddle.static.data(
                name='image', shape=[None] + image_shape, dtype='float32')
            predict = teacher_model.net(image, class_dim=class_dim)

    exe.run(t_startup)
    if not os.path.exists(args.teacher_pretrained_model):
        _download(
            'http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar',
            '.')
        _decompress('./ResNet50_vd_pretrained.tar')
    assert args.teacher_pretrained_model and os.path.exists(
        args.teacher_pretrained_model
    ), "teacher_pretrained_model should be set when teacher_model is not None."

    def if_exist(var):
        exist = os.path.exists(
            os.path.join(args.teacher_pretrained_model, var.name))
        if args.data == "cifar10" and (var.name == 'fc_0.w_0' or
                                       var.name == 'fc_0.b_0'):
            exist = False
        return exist

    paddle.static.load(teacher_program, args.teacher_pretrained_model, exe)

    data_name_map = {'image': 'image'}
    merge(teacher_program, student_program, data_name_map, place)

    with paddle.static.program_guard(student_program, s_startup):
        distill_loss = soft_label_loss("teacher_fc_0.tmp_0", "fc_0.tmp_0",
                                       student_program)
        loss = avg_cost + distill_loss
        lr, opt = create_optimizer(args)
        opt.minimize(loss)
    exe.run(s_startup)
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.fuse_all_reduce_ops = False
    parallel_main = paddle.static.CompiledProgram(
        student_program).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy)

    for epoch_id in range(args.num_epochs):
        for step_id, data in enumerate(train_loader):
            loss_1, loss_2, loss_3 = exe.run(
                parallel_main,
                feed=data,
                fetch_list=[loss.name, avg_cost.name, distill_loss.name])
            if step_id % args.log_period == 0:
                _logger.info(
                    "train_epoch {} step {} lr {:.6f}, loss {:.6f}, class loss {:.6f}, distill loss {:.6f}".
                    format(epoch_id, step_id,
                           lr.get_lr(), loss_1[0], loss_2[0], loss_3[0]))
            lr.step()
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
        if args.save_inference:
            paddle.fluid.io.save_inference_model(
                os.path.join("./saved_models", str(epoch_id)), ["image"],
                [out], exe, student_program)
        _logger.info("epoch {} top1 {:.6f}, top5 {:.6f}".format(
            epoch_id, np.mean(val_acc1s), np.mean(val_acc5s)))


def main():
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    paddle.enable_static()
    main()
