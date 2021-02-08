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
import paddle.vision.models as models
from utility import add_arguments, print_arguments
from paddle.jit import to_static

_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  64 * 4,                 "Minibatch size.")
add_arg('model',            str,  "mobilenet_v1",                "The target model.")
add_arg('data',             str, "imagenet",                 "Which data to use. 'mnist' or 'imagenet'")
add_arg('log_period',       int, 10,                 "Log period in batches.")
add_arg('test_period',      int, 10,                 "Test period in epoches.")
add_arg('checkpoint',   str, None,                "The path of checkpoint which is used for eval.")
add_arg('pruned_ratio',     float, None,         "The ratios to be pruned.")
add_arg('output_path',   str, None,                "The path of checkpoint which is used for eval.")
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


def export(args):

    paddle.set_device('cpu')
    test_reader = None
    if args.data == "cifar10":
        class_dim = 10
        image_shape = [3, 224, 224]
    elif args.data == "imagenet":
        class_dim = 1000
        image_shape = [3, 224, 224]
    else:
        raise ValueError("{} is not supported.".format(args.data))
    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    # model definition
    net = models.__dict__[args.model](pretrained=False, num_classes=class_dim)

    pruner = paddleslim.dygraph.L1NormFilterPruner(net, [1] + image_shape)
    params = get_pruned_params(args, net)
    ratios = {}
    for param in params:
        ratios[param] = args.pruned_ratio
    print("ratios: {}".format(ratios))
    pruner.prune_vars(ratios, [0])

    param_state_dict = paddle.load(args.checkpoint + ".pdparams")
    net.set_dict(param_state_dict)

    net.eval()
    model = to_static(
        net,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None] + image_shape, dtype='float32', name="image")
        ])
    paddle.jit.save(net, args.output_path)


def main():
    args = parser.parse_args()
    print_arguments(args)
    export(args)


if __name__ == '__main__':
    main()
