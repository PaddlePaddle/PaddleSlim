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
from paddleseg.cvlibs import Config as PaddleSegDataConfig
from paddleseg.utils import worker_init_fn

from paddleslim.auto_compression import AutoCompression
from paddleseg.core.infer import reverse_transform
from paddleseg.utils import metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
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
        '--save_dir',
        type=str,
        default=None,
        help="directory to save compressed model.")
    parser.add_argument(
        '--strategy_config',
        type=str,
        default=None,
        help="path of compression strategy config.")
    parser.add_argument(
        '--dataset_config',
        type=str,
        default=None,
        help="path of dataset config.")
    parser.add_argument(
        '--deploy_hardware',
        type=str,
        default=None,
        help="The hardware you want to deploy.")
    return parser.parse_args()


def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):

    nranks = paddle.distributed.ParallelEnv().local_rank

    batch_sampler = paddle.io.DistributedBatchSampler(
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

    for iter, (image, label) in enumerate(loader):
        paddle.enable_static()

        label = np.array(label).astype('int64')
        ori_shape = np.array(label).shape[-2:]

        image = np.array(image)
        logits = exe.run(compiled_test_program,
                         feed={test_feed_names[0]: image},
                         fetch_list=test_fetch_list,
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

        if nranks > 1:
            intersect_area_list = []
            pred_area_list = []
            label_area_list = []
            paddle.distributed.all_gather(intersect_area_list, intersect_area)
            paddle.distributed.all_gather(pred_area_list, pred_area)
            paddle.distributed.all_gather(label_area_list, label_area)

            # Some image has been evaluated and should be eliminated in last iter
            if (iter + 1) * nranks > len(eval_dataset):
                valid = len(eval_dataset) - iter * nranks
                intersect_area_list = intersect_area_list[:valid]
                pred_area_list = pred_area_list[:valid]
                label_area_list = label_area_list[:valid]

            for i in range(len(intersect_area_list)):
                intersect_area_all = intersect_area_all + intersect_area_list[i]
                pred_area_all = pred_area_all + pred_area_list[i]
                label_area_all = label_area_all + label_area_list[i]
        else:
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

    paddle.enable_static()
    return miou


def reader_wrapper(reader):
    def gen():
        for i, data in enumerate(reader()):
            imgs = np.array(data[0])
            yield {"x": imgs}

    return gen


if __name__ == '__main__':

    args = parse_args()
    paddle.enable_static()
    # step1: load dataset config and create dataloader
    data_cfg = PaddleSegDataConfig(args.dataset_config)
    train_dataset = data_cfg.train_dataset
    eval_dataset = data_cfg.val_dataset
    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        drop_last=True)
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=2,
        return_list=True,
        worker_init_fn=worker_init_fn)
    train_dataloader = reader_wrapper(train_loader)

    # step2: create and instance of AutoCompression
    ac = AutoCompression(
        model_dir=args.model_dir,
        model_filename=args.model_filename,
        params_filename=args.params_filename,
        save_dir=args.save_dir,
        config=args.strategy_config,
        train_dataloader=train_dataloader,
        eval_callback=eval_function,
        deploy_hardware=args.deploy_hardware)

    # step3: start the compression job
    ac.compress()
