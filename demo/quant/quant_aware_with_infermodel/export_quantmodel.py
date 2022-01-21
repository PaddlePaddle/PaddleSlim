# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import os
import sys
import math
import time
import numpy as np
import paddle
import logging
import argparse
import functools

sys.path[0] = os.path.join(
    os.path.dirname("__file__"), os.path.pardir, os.path.pardir)
sys.path[1] = os.path.join(
    os.path.dirname("__file__"), os.path.pardir, os.path.pardir, os.path.pardir)
from paddleslim.common import get_logger
from paddleslim.quant import export_quant_infermodel
from utility import add_arguments, print_arguments
import imagenet_reader as reader
_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('use_gpu',                     bool,   True,      "Whether to use GPU or not.")
add_arg('batch_size',                  int,    4,         "train batch size.")
add_arg('num_epoch',                   int,    1,         "train epoch num.")
add_arg('save_iter_step',              int,    1,         "save train checkpoint every save_iter_step iter num.")
add_arg('learning_rate',               float,  0.0001,    "learning rate.")
add_arg('weight_decay',                float,  0.00004,   "weight decay.")
add_arg('use_pact',                    bool,   True,      "whether use pact quantization.")
add_arg('checkpoint_path',             str,    None,      "model dir to save quanted model checkpoints")
add_arg('model_path_prefix',           str,    None,      "storage directory of model + model name (excluding suffix)")
add_arg('teacher_model_path_prefix',   str,    None,      "storage directory of teacher model + teacher model name (excluding suffix)")
add_arg('distill_node_name_list',      str,    None,      "distill node name list", nargs="+")
add_arg('checkpoint_filename',         str,    None,      "checkpoint filename to export inference model")
add_arg('export_inference_model_path_prefix',   str,    None,      "inference model export path prefix")

def export(args):
    place = paddle.CUDAPlace(0) if args.use_gpu else paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    quant_config = {
        'weight_quantize_type': 'channel_wise_abs_max',
        'activation_quantize_type': 'moving_average_abs_max',
        'not_quant_pattern': ['skip_quant'],
        'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul']
    }
    train_config={
        "num_epoch": args.num_epoch, # training epoch num
        "max_iter": -1,
        "save_iter_step": args.save_iter_step,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "use_pact": args.use_pact,
        "quant_model_ckpt_path":args.checkpoint_path,
        "teacher_model_path_prefix": args.teacher_model_path_prefix,
        "model_path_prefix": args.model_path_prefix,
        "distill_node_pair": args.distill_node_name_list
    }

    export_quant_infermodel(exe, place,
        scope=None,
        quant_config=quant_config,
        train_config=train_config,
        checkpoint_path=os.path.join(args.checkpoint_path, args.checkpoint_filename),
        export_inference_model_path_prefix=args.export_inference_model_path_prefix)

def main():
    args = parser.parse_args()
    args.use_pact = bool(args.use_pact)
    print_arguments(args)
    export(args)


if __name__ == '__main__':
    paddle.enable_static()
    main()
