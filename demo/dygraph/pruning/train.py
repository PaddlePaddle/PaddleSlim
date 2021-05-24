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
sys.path.append(
    os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir))
import paddleslim
from paddleslim.common import get_logger
from paddleslim.analysis import dygraph_flops as flops
import paddle.vision.models as models
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
parser.add_argument('--step_epochs', nargs='+', type=int, default=[30, 60, 90], help="piecewise decay step")
add_arg('data',             str, "mnist",                 "Which data to use. 'mnist' or 'imagenet'")
add_arg('log_period',       int, 10,                 "Log period in batches.")
add_arg('test_period',      int, 10,                 "Test period in epoches.")
add_arg('model_path',       str, "./models",         "The path to save model.")
add_arg('pruned_ratio',     float, None,         "The ratios to be pruned.")
add_arg('criterion',        str, "l1_norm",         "The prune criterion to be used, support l1_norm and batch_norm_scale.")
add_arg('use_gpu',   bool, True,                "Whether to GPUs.")
add_arg('checkpoint',   str, None,                "The path of checkpoint which is used for resume training.")
# yapf: enable

model_list = models.__all__


def get_pruned_params(args, model):
    params = []
    if args.model == "mobilenet_v1":
        skip_vars = ['linear_0.b_0',
                     'conv2d_0.w_0']  # skip the first conv2d and last linear
        for sublayer in model.sublayers():
            for param in sublayer.parameters(include_sublayers=False):
                if isinstance(
                        sublayer, paddle.nn.Conv2D
                ) and sublayer._groups == 1 and param.name not in skip_vars:
                    params.append(param.name)
    elif args.model == "mobilenet_v2":
        for sublayer in model.sublayers():
            for param in sublayer.parameters(include_sublayers=False):
                if isinstance(sublayer, paddle.nn.Conv2D):
                    params.append(param.name)
        return params
    elif args.model == "resnet34":
        for sublayer in model.sublayers():
            for param in sublayer.parameters(include_sublayers=False):
                if isinstance(sublayer, paddle.nn.Conv2D):
                    params.append(param.name)
        return params
    else:
        raise NotImplementedError(
            "Current demo only support for mobilenet_v1, mobilenet_v2, resnet34")
    return params


def piecewise_decay(args, parameters, steps_per_epoch):
    bd = [steps_per_epoch * e for e in args.step_epochs]
    lr = [args.lr * (0.1**i) for i in range(len(bd) + 1)]
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(boundaries=bd, values=lr)

    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay),
        parameters=parameters)
    return optimizer


def cosine_decay(args, parameters, steps_per_epoch):
    learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.lr, T_max=args.num_epochs * steps_per_epoch)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay),
        parameters=parameters)
    return optimizer


def create_optimizer(args, parameters, steps_per_epoch):
    if args.lr_strategy == "piecewise_decay":
        return piecewise_decay(args, parameters, steps_per_epoch)
    elif args.lr_strategy == "cosine_decay":
        return cosine_decay(args, parameters, steps_per_epoch)


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
            "data/ILSVRC2012",
            mode='train',
            image_size=224,
            resize_short_size=256)

        val_dataset = ImageNetDataset(
            "data/ILSVRC2012",
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
    inputs = [Input([None] + image_shape, 'float32', name='image')]
    labels = [Input([None, 1], 'int64', name='label')]

    # model definition
    net = models.__dict__[args.model](pretrained=pretrain,
                                      num_classes=class_dim)

    _logger.info("FLOPs before pruning: {}GFLOPs".format(
        flops(net, [1] + image_shape) / 1000))
    net.eval()
    if args.criterion == 'fpgm':
        pruner = paddleslim.dygraph.FPGMFilterPruner(net, [1] + image_shape)
    elif args.criterion == 'l1_norm':
        pruner = paddleslim.dygraph.L1NormFilterPruner(net, [1] + image_shape)

    params = get_pruned_params(args, net)
    ratios = {}
    for param in params:
        ratios[param] = args.pruned_ratio
    plan = pruner.prune_vars(ratios, [0])

    _logger.info("FLOPs after pruning: {}GFLOPs; pruned ratio: {}".format(
        flops(net, [1] + image_shape) / 1000, plan.pruned_flops))

    for param in net.parameters():
        if "conv2d" in param.name:
            print("{}\t{}".format(param.name, param.shape))

    net.train()
    model = paddle.Model(net, inputs, labels)
    steps_per_epoch = int(np.ceil(len(train_dataset) * 1. / args.batch_size))
    opt = create_optimizer(args, net.parameters(), steps_per_epoch)
    model.prepare(
        opt, paddle.nn.CrossEntropyLoss(), paddle.metric.Accuracy(topk=(1, 5)))
    if args.checkpoint is not None:
        model.load(args.checkpoint)
    model.fit(train_data=train_dataset,
              eval_data=val_dataset,
              epochs=args.num_epochs,
              batch_size=args.batch_size // ParallelEnv().nranks,
              verbose=1,
              save_dir=args.model_path,
              num_workers=8)


def main():
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()
