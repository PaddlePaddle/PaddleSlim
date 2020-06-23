# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import ast
import argparse
import functools

import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
import reader
from model_search import Network
from paddleslim.nas.darts import DARTSearch
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('log_freq',          int,   50,              "Log frequency.")
add_arg('use_multiprocess',  bool,  True,            "Whether use multiprocess reader.")
add_arg('batch_size',        int,   64,              "Minibatch size.")
add_arg('learning_rate',     float, 0.025,            "The start learning rate.")
add_arg('momentum',          float, 0.9,             "Momentum.")
add_arg('use_gpu',           bool,  True,            "Whether use GPU.")
add_arg('epochs',            int,   50,              "Epoch number.")
add_arg('init_channels',     int,   16,              "Init channel number.")
add_arg('layers',            int,   8,               "Total number of layers.")
add_arg('class_num',         int,   10,              "Class number of dataset.")
add_arg('trainset_num',      int,   50000,           "images number of trainset.")
add_arg('model_save_dir',    str,   'search_cifar', "The path to save model.")
add_arg('grad_clip',         float, 5,               "Gradient clipping.")
add_arg('arch_learning_rate',float, 3e-4,            "Learning rate for arch encoding.")
add_arg('method',            str,   'DARTS',         "The search method you would like to use")
add_arg('epochs_no_archopt', int,   0,               "Epochs not optimize the arch params")
add_arg('cutout_length',     int,   16,              "Cutout length.")
add_arg('cutout',            ast.literal_eval,  False, "Whether use cutout.")
add_arg('unrolled',          ast.literal_eval,  False, "Use one-step unrolled validation loss")
add_arg('use_data_parallel', ast.literal_eval,  False, "The flag indicating whether to use data parallel mode to train the model.")
# yapf: enable


def main(args):
    if not args.use_gpu:
        place = fluid.CPUPlace()
    elif not args.use_data_parallel:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)

    train_reader, valid_reader = reader.train_search(
        batch_size=args.batch_size,
        train_portion=0.5,
        is_shuffle=True,
        args=args)

    with fluid.dygraph.guard(place):
        model = Network(args.init_channels, args.class_num, args.layers,
                        args.method)
        searcher = DARTSearch(
            model,
            train_reader,
            valid_reader,
            place,
            learning_rate=args.learning_rate,
            batchsize=args.batch_size,
            num_imgs=args.trainset_num,
            arch_learning_rate=args.arch_learning_rate,
            unrolled=args.unrolled,
            num_epochs=args.epochs,
            epochs_no_archopt=args.epochs_no_archopt,
            use_multiprocess=args.use_multiprocess,
            use_data_parallel=args.use_data_parallel,
            save_dir=args.model_save_dir,
            log_freq=args.log_freq)
        searcher.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    main(args)
