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
import paddle
from tqdm import tqdm
from post_process import YOLOv6PostProcess, coco_metric
from dataset import COCOValDataset, COCOTrainDataset
from paddleslim.common import load_config, load_onnx_model
from paddleslim.quant.analysis import AnalysisQuant


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of analysis config.",
        required=True)
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")
    return parser


def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):
    bboxes_list, bbox_nums_list, image_id_list = [], [], []
    with tqdm(
            total=len(val_loader),
            bar_format='Evaluation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
            ncols=80) as t:
        for data in val_loader:
            data_all = {k: np.array(v) for k, v in data.items()}
            outs = exe.run(compiled_test_program,
                           feed={test_feed_names[0]: data_all['image']},
                           fetch_list=test_fetch_list,
                           return_numpy=False)
            res = {}
            postprocess = YOLOv6PostProcess(
                score_threshold=0.001, nms_threshold=0.65, multi_label=True)
            res = postprocess(np.array(outs[0]), data_all['scale_factor'])
            bboxes_list.append(res['bbox'])
            bbox_nums_list.append(res['bbox_num'])
            image_id_list.append(np.array(data_all['im_id']))
            t.update()
    map_res = coco_metric(anno_file, bboxes_list, bbox_nums_list, image_id_list)
    return map_res[0]


def main():

    global config
    config = load_config(FLAGS.config_path)

    dataset = COCOTrainDataset(
        dataset_dir=config['dataset_dir'],
        image_dir=config['val_image_dir'],
        anno_path=config['val_anno_path'])
    data_loader = paddle.io.DataLoader(
        dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0)

    global val_loader
    dataset = COCOValDataset(
        dataset_dir=config['dataset_dir'],
        image_dir=config['val_image_dir'],
        anno_path=config['val_anno_path'])
    global anno_file
    anno_file = dataset.ann_file
    val_loader = paddle.io.DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    load_onnx_model(config["model_dir"])
    inference_model_path = config["model_dir"].rstrip().rstrip(
        '.onnx') + '_infer'
    analyzer = AnalysisQuant(
        model_dir=inference_model_path,
        model_filename='model.pdmodel',
        params_filename='model.pdiparams',
        eval_function=eval_function,
        quantizable_op_type=config['quantizable_op_type'],
        weight_quantize_type=config['weight_quantize_type'],
        activation_quantize_type=config['activation_quantize_type'],
        is_full_quantize=config['is_full_quantize'],
        data_loader=data_loader,
        batch_size=config['batch_size'],
        save_dir=config['save_dir'], )
    analyzer.analysis()


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()

    assert FLAGS.devices in ['cpu', 'gpu', 'xpu', 'npu']
    paddle.set_device(FLAGS.devices)

    main()
