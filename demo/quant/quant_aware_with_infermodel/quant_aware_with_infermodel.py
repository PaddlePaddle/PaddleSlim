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
from paddleslim.quant import quant_aware_with_infermodel
from utility import add_arguments, print_arguments
import imagenet_reader as reader
_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('use_gpu',                     bool,   True,      "whether to use GPU or not.")
add_arg('batch_size',                  int,    1,         "train batch size.")
add_arg('num_epoch',                   int,    1,         "train epoch num.")
add_arg('save_iter_step',              int,    1,         "save train checkpoint every save_iter_step iter num.")
add_arg('learning_rate',               float,  0.0001,    "learning rate.")
add_arg('weight_decay',                float,  0.00004,   "weight decay.")
add_arg('use_pact',                    bool,   True,      "whether use pact quantization.")
add_arg('checkpoint_path',             str,    None,      "model dir to save quanted model checkpoints")
add_arg('model_path_prefix',           str,    None,      "storage directory of model + model name (excluding suffix)")
add_arg('teacher_model_path_prefix',   str,    None,      "storage directory of teacher model + teacher model name (excluding suffix)")
add_arg('distill_node_name_list',      str,    None,      "distill node name list", nargs="+")


DATA_DIR = "../../data/ILSVRC2012/"
def eval(exe, place, compiled_test_program, test_feed_names, test_fetch_list):
    val_reader = paddle.batch(reader.val(), batch_size=1)
    image = paddle.static.data(
        name='x', shape=[None, 3, 224, 224], dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')

    results = []
    for batch_id, data in enumerate(val_reader()):
        # top1_acc, top5_acc
        if len(test_feed_names) == 1:
            # eval "infer model", which input is image, output is classification probability
            image = data[0][0].reshape((1, 3, 224, 224))
            label = [[d[1]] for d in data]
            pred = exe.run(compiled_test_program,
                           feed={test_feed_names[0]: image},
                           fetch_list=test_fetch_list)
            pred = np.array(pred[0])
            label = np.array(label)
            sort_array = pred.argsort(axis=1)
            top_1_pred = sort_array[:, -1:][:, ::-1]
            top_1 = np.mean(label == top_1_pred)
            top_5_pred = sort_array[:, -5:][:, ::-1]
            acc_num = 0
            for i in range(len(label)):
                if label[i][0] in top_5_pred[i]:
                    acc_num += 1
            top_5 = float(acc_num) / len(label)
            results.append([top_1, top_5])
        else:
            # eval "eval model", which inputs are image and label, output is top1 and top5 accuracy
            image = data[0][0].reshape((1, 3, 224, 224))
            label = [[d[1]] for d in data]
            result = exe.run(compiled_test_program,
                             feed={
                                 test_feed_names[0]: image,
                                 test_feed_names[1]: label
                             },
                             fetch_list=test_fetch_list)
            result = [np.mean(r) for r in result]
            results.append(result)
    result = np.mean(np.array(results), axis=0)
    return result

def quantize(args):
    place = paddle.CUDAPlace(0) if args.use_gpu else paddle.CPUPlace()
    #place = paddle.CPUPlace()

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
    def test_callback(compiled_test_program, feed_names, fetch_list, checkpoint_name):
        ret = eval(exe, place, compiled_test_program, feed_names, fetch_list)
        print("{0} top1_acc/top5_acc= {1}".format(checkpoint_name, ret))

    train_reader = paddle.batch(reader.train(), batch_size=args.batch_size)
    def train_reader_wrapper():
        def gen():
            for i, data in enumerate(train_reader()):
                imgs = np.float32([item[0] for item in data])
                yield {"x":imgs}
        return gen
    quant_aware_with_infermodel(
        exe,
        place,
        scope=None,
        train_reader=train_reader_wrapper(),
        quant_config=quant_config,
        train_config=train_config,
        test_callback=test_callback)

def main():
    args = parser.parse_args()
    args.use_pact=bool(args.use_pact)
    print("args.use_pact", args.use_pact)
    print_arguments(args)
    quantize(args)


if __name__ == '__main__':
    paddle.enable_static()
    main()
