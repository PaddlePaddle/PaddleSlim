# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
sys.path[0] = os.path.join(
    os.path.dirname("__file__"), os.path.pardir, os.path.pardir)
import argparse
import functools
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset, BatchSampler, DataLoader
import imagenet_reader as reader
from paddleslim.auto_compression.config_helpers import load_config as load_slim_config


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)


def eval_reader(data_dir, batch_size):
    val_reader = paddle.batch(
        reader.val(data_dir=data_dir), batch_size=batch_size)
    return val_reader


def eval():
    devices = paddle.device.get_device().split(':')[0]
    places = paddle.device._convert_to_place(devices)
    exe = paddle.static.Executor(places)
    val_program, feed_target_names, fetch_targets = paddle.static.load_inference_model(
        global_config["model_dir"],
        exe,
        model_filename=global_config["model_filename"],
        params_filename=global_config["params_filename"])
    print('Loaded model from: {}'.format(global_config["model_dir"]))

    val_reader = eval_reader(data_dir, batch_size=global_config['batch_size'])
    image = paddle.static.data(
        name=global_config['input_name'],
        shape=[None, 3, 224, 224],
        dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    results = []
    print('Evaluating... It will take a while. Please wait...')
    for batch_id, data in enumerate(val_reader()):
        # top1_acc, top5_acc
        image = np.array([[d[0]] for d in data])
        image = image.reshape((len(data), 3, 224, 224))
        label = [[d[1]] for d in data]
        pred = exe.run(val_program,
                       feed={feed_target_names[0]: image},
                       fetch_list=fetch_targets)
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
    result = np.mean(np.array(results), axis=0)
    return result[0]


def main():
    global global_config
    all_config = load_slim_config(args.config_path)
    assert "Global" in all_config, f"Key 'Global' not found in config file. \n{all_config}"
    global_config = all_config["Global"]
    global data_dir
    data_dir = global_config['data_dir']
    result = eval()
    print('Eval Top1:', result)


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    args = parser.parse_args()
    main()
