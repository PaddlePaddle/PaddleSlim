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
import argparse
import functools
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from imagenet_reader import ImageNetDataset
from paddleslim.auto_compression.config_helpers import load_config as load_slim_config


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default='./image_classification/configs/eval.yaml',
        help="path of compression strategy config.")
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./MobileNetV1_infer',
        help='model directory')
    return parser


def eval_reader(data_dir, batch_size, crop_size, resize_size):
    val_reader = ImageNetDataset(
        mode='val',
        data_dir=data_dir,
        crop_size=crop_size,
        resize_size=resize_size)
    val_loader = DataLoader(
        val_reader,
        batch_size=global_config['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=0)
    return val_loader


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

    val_loader = eval_reader(
        data_dir,
        batch_size=global_config['batch_size'],
        crop_size=img_size,
        resize_size=resize_size)
    results = []
    print('Evaluating...')
    for batch_id, (image, label) in enumerate(val_loader):
        image = np.array(image)
        label = np.array(label).astype('int64')
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


def main(args):
    global global_config
    all_config = load_slim_config(args.config_path)
    global_config = all_config

    global data_dir
    data_dir = global_config['data_dir']
    if args.model_dir != global_config['model_dir']:
        config['model_dir'] = args.model_dir

    global img_size, resize_size
    img_size = int(global_config[
        'img_size']) if 'img_size' in global_config else 224
    resize_size = int(global_config[
        'resize_size']) if 'resize_size' in global_config else 256

    result = eval()
    print('Eval Top1:', result)


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    args = parser.parse_args()
    main(args)
