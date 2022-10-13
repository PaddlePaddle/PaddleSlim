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
import logging
import numpy as np
import argparse
from tqdm import tqdm
import paddle
from paddleslim.common import load_config as load_slim_config
from paddleslim.common import get_logger
from paddleslim.auto_compression import AutoCompression
from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric

logger = get_logger(__name__, level=logging.INFO)


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
        default='./ch_PP-OCRv3_det_infer',
        help='model directory')
    return parser


extra_input_models = [
    "SRN", "NRTR", "SAR", "SEED", "SVTR", "VisionLAN", "RobustScanner"
]


def sample_generator(loader):
    def __reader__():
        for indx, data in enumerate(loader):
            images = np.array(data[0])
            yield images

    return __reader__


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

    val_loader = build_dataloader(all_config, 'Eval', devices, logger)
    post_process_class = build_post_process(all_config['PostProcess'],
                                            global_config)
    eval_class = build_metric(all_config['Metric'])
    model_type = global_config['model_type']
    extra_input = True if global_config[
        'algorithm'] in extra_input_models else False

    with tqdm(
            total=len(val_loader),
            bar_format='Evaluation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
            ncols=80) as t:
        for batch_id, batch in enumerate(val_loader):
            images = batch[0]
            if extra_input:
                preds = exe.run(
                    val_program,
                    feed={feed_target_names[0]: images,
                          'data': batch[1:]},
                    fetch_list=fetch_targets)
            else:
                preds = exe.run(val_program,
                                feed={feed_target_names[0]: images},
                                fetch_list=fetch_targets)

            batch_numpy = []
            for item in batch:
                batch_numpy.append(np.array(item))

            if model_type == 'det':
                preds_map = {'maps': preds[0]}
                post_result = post_process_class(preds_map, batch_numpy[1])
                eval_class(post_result, batch_numpy)
            elif model_type == 'rec':
                post_result = post_process_class(preds[0], batch_numpy[1])
                eval_class(post_result, batch_numpy)

            t.update()
        metric = eval_class.get_metric()
    logger.info('metric eval ***************')
    for k, v in metric.items():
        logger.info('{}:{}'.format(k, v))
    return metric


def main():
    global all_config, global_config
    all_config = load_slim_config(args.config_path)
    global_config = all_config["Global"]
    eval()


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    args = parser.parse_args()
    main()
