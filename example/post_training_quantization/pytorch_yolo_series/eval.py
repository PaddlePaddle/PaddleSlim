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
import numpy as np
import argparse
from tqdm import tqdm
import paddle
from paddleslim.common import load_config as load_slim_config
from paddleslim.common import load_inference_model
from post_process import YOLOPostProcess, coco_metric
from dataset import COCOValDataset


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--batch_size', type=int, default=1, help="Batch size of model input.")
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")

    return parser


def eval():

    place = paddle.CUDAPlace(0) if FLAGS.devices == 'gpu' else paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    val_program, feed_target_names, fetch_targets = load_inference_model(
        config["model_dir"].rstrip('/'),
        exe,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams")

    bboxes_list, bbox_nums_list, image_id_list = [], [], []
    with tqdm(
            total=len(val_loader),
            bar_format='Evaluation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
            ncols=80) as t:
        for data in val_loader:
            data_all = {k: np.array(v) for k, v in data.items()}
            outs = exe.run(val_program,
                           feed={feed_target_names[0]: data_all['image']},
                           fetch_list=fetch_targets,
                           return_numpy=False)
            postprocess = YOLOPostProcess(
                score_threshold=0.001, nms_threshold=0.65, multi_label=True)
            res = postprocess(np.array(outs[0]), data_all['scale_factor'])
            bboxes_list.append(res['bbox'])
            bbox_nums_list.append(res['bbox_num'])
            image_id_list.append(np.array(data_all['im_id']))
            t.update()

    coco_metric(anno_file, bboxes_list, bbox_nums_list, image_id_list)


def main():
    global config
    config = load_slim_config(FLAGS.config_path)

    global val_loader
    dataset = COCOValDataset(
        dataset_dir=config['dataset_dir'],
        image_dir=config['val_image_dir'],
        anno_path=config['val_anno_path'])
    global anno_file
    anno_file = dataset.ann_file
    val_loader = paddle.io.DataLoader(
        dataset, batch_size=FLAGS.batch_size, drop_last=True)

    eval()


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()

    assert FLAGS.devices in ['cpu', 'gpu', 'xpu', 'npu']
    paddle.set_device(FLAGS.devices)

    main()
