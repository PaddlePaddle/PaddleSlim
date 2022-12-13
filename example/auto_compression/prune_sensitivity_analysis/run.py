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
import pickle
import functools
from functools import partial
import math
from tqdm import tqdm

import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
import paddleslim
from imagenet_reader import ImageNetDataset
from paddleslim.common import load_config as load_slim_config
from paddleslim.auto_compression.analysis import analysis_prune


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--analysis_file',
        type=str,
        default='sensitivity_0.data',
        help="directory to save compressed model.")
    parser.add_argument(
        '--pruned_ratios',
        nargs='+',
        type=float,
        default=[0.1, 0.2, 0.3, 0.4],
        help="The ratios to be pruned when compute sensitivity.")
    parser.add_argument(
        '--target_loss',
        type=float,
        default=0.2,
        help="use the target loss to get prune ratio of each parameter")

    return parser


def eval_reader(data_dir, batch_size, crop_size, resize_size, place=None):
    val_reader = ImageNetDataset(
        mode='val',
        data_dir=data_dir,
        crop_size=crop_size,
        resize_size=resize_size)
    val_loader = DataLoader(
        val_reader,
        places=[place] if place is not None else None,
        batch_size=global_config['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=0)
    return val_loader


def eval_function(compiled_test_program, exe, test_feed_names, test_fetch_list):
    val_loader = eval_reader(
        global_config['data_dir'],
        batch_size=global_config['batch_size'],
        crop_size=img_size,
        resize_size=resize_size)

    results = []
    with tqdm(
            total=len(val_loader),
            bar_format='Evaluation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
            ncols=80) as t:
        for batch_id, (image, label) in enumerate(val_loader):
            # top1_acc, top5_acc
            if len(test_feed_names) == 1:
                image = np.array(image)
                label = np.array(label).astype('int64')
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
                image = np.array(image)
                label = np.array(label).astype('int64')
                result = exe.run(compiled_test_program,
                                 feed={
                                     test_feed_names[0]: image,
                                     test_feed_names[1]: label
                                 },
                                 fetch_list=test_fetch_list)
                result = [np.mean(r) for r in result]
                results.append(result)
            t.update()
    result = np.mean(np.array(results), axis=0)
    return result[0]


def main():
    global global_config
    all_config = load_slim_config(args.config_path)

    assert "Global" in all_config, f"Key 'Global' not found in config file. \n{all_config}"
    global_config = all_config["Global"]

    global img_size, resize_size
    img_size = global_config['img_size'] if 'img_size' in global_config else 224
    resize_size = global_config[
        'resize_size'] if 'resize_size' in global_config else 256

    analysis_prune(eval_function, global_config['model_dir'],
                   global_config['model_filename'],
                   global_config['params_filename'], args.analysis_file,
                   args.pruned_ratios, args.target_loss)


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    args = parser.parse_args()
    main()
