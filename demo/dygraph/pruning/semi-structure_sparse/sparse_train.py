from __future__ import division
from __future__ import print_function
import os
import sys
import logging
import paddle
import argparse
import functools
import math
import time
import numpy as np
sys.path.append(os.path.join(os.path.dirname("__file__"), os.path.pardir))
sys.path.append(
    os.path.join(
        os.path.dirname("__file__"), os.path.pardir, os.path.pardir,
        os.path.pardir))
import paddleslim
from paddleslim.common import get_logger
from paddleslim.analysis import dygraph_flops as flops
import models
from utility import add_arguments, print_arguments
import paddle.vision.transforms as T
from paddle.static import InputSpec as Input
from imagenet import ImageNetDataset
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler
from paddle.distributed import ParallelEnv

_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  64 * 4,                 "Minibatch size.")
add_arg('model',            str,  "MobileNet",                "The target model.")
add_arg('lr',               float,  0.1,               "The learning rate used to fine-tune pruned model.")
add_arg('lr_strategy',      str,  "piecewise_decay",   "The learning rate decay strategy.")
add_arg('l2_decay',         float,  3e-5,               "The l2_decay parameter.")
add_arg('momentum_rate',    float,  0.9,               "The value of momentum_rate.")
add_arg('num_epochs',       int,  120,               "The number of total epochs.")
add_arg('target_rate',       int, 12,               "The number of weights would been pruned per group finally")
parser.add_argument('--step_epochs', nargs='+', type=int, default=[30, 60, 90], help="piecewise decay step")
add_arg('data',             str, "mnist",                 "Which data to use. 'mnist' or 'imagenet'")
add_arg('model_path',       str, "./models",         "The path to save model.")
add_arg('use_gpu',   bool, True,                "Whether to GPUs.")
add_arg('checkpoint',   str, None,                "The path of checkpoint which is used for resume training.")
# yapf: enable

model_list = models.__all__


def piecewise_decay(args, parameters, steps_per_epoch):
    bd = [steps_per_epoch * e for e in args.step_epochs]
    lr = [args.lr * (0.1**i) for i in range(len(bd) + 1)]
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(boundaries=bd, values=lr)

    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay),
        parameters=parameters)
    return optimizer, lr


def cosine_decay(args, parameters, steps_per_epoch):
    learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.lr, T_max=args.num_epochs * steps_per_epoch)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay),
        parameters=parameters)
    return optimizer, learning_rate


def create_optimizer(args, parameters, steps_per_epoch):
    if args.lr_strategy == "piecewise_decay":
        return piecewise_decay(args, parameters, steps_per_epoch)
    elif args.lr_strategy == "cosine_decay":
        return cosine_decay(args, parameters, steps_per_epoch)


def train(epoch, model, optim, lr, train_loader):
    for batch_id, data in enumerate(train_loader()):
        img = data[0]
        label = data[1]
        loss_fn = paddle.nn.CrossEntropyLoss()
        predicts = model(img)
        loss = loss_fn(predicts, label)
        acc = paddle.metric.accuracy(predicts, label)
        loss.backward()
        optim.step()
        optim.clear_grad()
        lr.step()
        if (batch_id + 1) % 50 == 0:
            _logger.info(
                "epoch: {}, batch_id: {}, lr is: {}, loss is: {}, acc is: {}".
                format(epoch, batch_id + 1,
                       round(lr.get_lr(), 3), loss.numpy()[0], acc.numpy()[0]))


def evaluate(model, val_loader):
    accuracy = []
    for batch_id, data in enumerate(val_loader()):
        img = data[0]
        label = data[1]
        predicts = model(img)
        acc = paddle.metric.accuracy(predicts, label)
        accuracy.append(float(acc))
        if (batch_id + 1) % 50 == 0:
            _logger.info("batch_id: {}, acc is: {}".format(batch_id + 1,
                                                           acc.numpy()))
    _logger.info('Final validation acc {}'.format(np.mean(accuracy)))


def compress(args):

    paddle.set_device('gpu' if args.use_gpu else 'cpu')
    train_reader = None
    test_reader = None
    if args.data == "cifar10":

        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])

        train_dataset = paddle.vision.datasets.Cifar10(
            mode="train", backend="cv2", transform=transform)
        val_dataset = paddle.vision.datasets.Cifar10(
            mode="test", backend="cv2", transform=transform)
        class_dim = 10
        image_shape = [3, 32, 32]
        pretrain = False
    elif args.data == "imagenet":

        train_dataset = ImageNetDataset(
            "/root/data/ILSVRC2012",
            mode='train',
            image_size=224,
            resize_short_size=256)

        val_dataset = ImageNetDataset(
            "/root/data/ILSVRC2012",
            mode='val',
            image_size=224,
            resize_short_size=256)

        class_dim = 1000
        image_shape = [3, 224, 224]
        pretrain = True
    else:
        raise ValueError("{} is not supported.".format(args.data))
    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    net = models.__dict__[args.model](class_dim=class_dim)

    net.train()

    steps_per_epoch = int(np.ceil(len(train_dataset) * 1. / args.batch_size))
    opt, lr = create_optimizer(args, net.parameters(), steps_per_epoch)

    if args.checkpoint is not None:
        model.load(args.checkpoint)

    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size=args.batch_size // ParallelEnv().nranks,
        shuffle=True)
    val_loader = paddle.io.DataLoader(
        val_dataset,
        batch_size=args.batch_size // ParallelEnv().nranks,
        shuffle=False)

    ########## Compute Targeted Dropout Schedule ##########
    target_rate_dict = []
    drop_rate_dict = []
    cur_drop_rate = 0.5
    cur_target_rate = 0
    step = args.num_epochs / 2 / (args.target_rate + 1)
    rise_point = [int(x * step) for x in range(args.target_rate + 1)]
    for i in range(0, args.num_epochs):
        if i >= args.num_epochs / 2:
            cur_drop_rate += 0.5 / (args.num_epochs / 2)
        if i in rise_point[1:]:
            cur_target_rate += 1
        drop_rate_dict.append(cur_drop_rate)
        target_rate_dict.append(cur_target_rate)
    #######################################################

    for epoch in range(args.num_epochs):
        Real_sparse = []
        for n, m in net.named_sublayers():
            if 'get_infer_sparsity' in dir(m):
                m.set_target_rate(target_rate_dict[epoch])
                m.set_drop_rate(drop_rate_dict[epoch])
                Real_sparse.append(m.get_real_sparsity(1e-10))
                Infer_sparse = m.get_infer_sparsity()
        _logger.info(
            'Epoch {}: Target Rate: {}, Drop Rate: {}, Infer Sparse: {}'.format(
                epoch, target_rate_dict[epoch], drop_rate_dict[
                    epoch], Infer_sparse))
        _logger.info('Real_sparse: ' + str(Real_sparse))

        train(epoch, net, opt, lr, train_loader)
        evaluate(net, val_loader)


def main():
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()
