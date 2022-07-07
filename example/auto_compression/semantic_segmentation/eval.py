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
import argparse
import random
import paddle
import numpy as np
from tqdm import tqdm
from paddleseg.cvlibs import Config as PaddleSegDataConfig
from paddleseg.utils import worker_init_fn

from paddleseg.core.infer import reverse_transform
from paddleseg.utils import metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')
    parser.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help="inference model directory.")
    parser.add_argument(
        '--model_filename',
        type=str,
        default=None,
        help="inference model filename.")
    parser.add_argument(
        '--params_filename',
        type=str,
        default=None,
        help="inference params filename.")
    parser.add_argument(
        '--dataset_config',
        type=str,
        default=None,
        help="path of dataset config.")
    return parser.parse_args()


def eval(args):
    exe = paddle.static.Executor(paddle.CUDAPlace(0))
    inference_program, feed_target_names, fetch_targets = paddle.static.load_inference_model(
        args.model_dir,
        exe,
        model_filename=args.model_filename,
        params_filename=args.params_filename)

    data_cfg = PaddleSegDataConfig(args.dataset_config)
    eval_dataset = data_cfg.val_dataset

    batch_sampler = paddle.io.BatchSampler(
        eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=1,
        return_list=True, )

    total_iters = len(loader)
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0

    print("Start evaluating (total_samples: {}, total_iters: {})...".format(
        len(eval_dataset), total_iters))

    for (image, label) in tqdm(loader):
        label = np.array(label).astype('int64')
        ori_shape = np.array(label).shape[-2:]
        image = np.array(image)
        logits = exe.run(inference_program,
                         feed={feed_target_names[0]: image},
                         fetch_list=fetch_targets,
                         return_numpy=True)

        paddle.disable_static()
        logit = logits[0]

        logit = reverse_transform(
            paddle.to_tensor(logit),
            ori_shape,
            eval_dataset.transforms.transforms,
            mode='bilinear')
        pred = paddle.to_tensor(logit)
        if len(
                pred.shape
        ) == 4:  # for humanseg model whose prediction is distribution but not class id
            pred = paddle.argmax(pred, axis=1, keepdim=True, dtype='int32')

        intersect_area, pred_area, label_area = metrics.calculate_area(
            pred,
            paddle.to_tensor(label),
            eval_dataset.num_classes,
            ignore_index=eval_dataset.ignore_index)
        intersect_area_all = intersect_area_all + intersect_area
        pred_area_all = pred_area_all + pred_area
        label_area_all = label_area_all + label_area

    class_iou, miou = metrics.mean_iou(intersect_area_all, pred_area_all,
                                       label_area_all)
    class_acc, acc = metrics.accuracy(intersect_area_all, pred_area_all)
    kappa = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)
    class_dice, mdice = metrics.dice(intersect_area_all, pred_area_all,
                                     label_area_all)

    infor = "[EVAL] #Images: {} mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} Dice: {:.4f}".format(
        len(eval_dataset), miou, acc, kappa, mdice)
    print(infor)


if __name__ == '__main__':
    rank_id = paddle.distributed.get_rank()
    place = paddle.CUDAPlace(rank_id)
    args = parse_args()
    paddle.enable_static()
    eval(args)
