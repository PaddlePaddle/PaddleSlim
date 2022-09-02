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
import numpy as np
import cv2
import time
import sys
import argparse
import yaml
from tqdm import tqdm

from utils import preprocess, postprocess
import paddle
from paddle.inference import create_predictor
from paddleslim.common import load_config
from paddle.io import DataLoader
from imagenet_reader import ImageNetDataset


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default='./image_classification/configs/infer.yaml',
        help='config file path')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./MobileNetV1_infer',
        help='model directory')
    parser.add_argument(
        '--use_fp16', type=bool, default=False, help='Whether to use fp16')
    parser.add_argument(
        '--use_int8', type=bool, default=False, help='Whether to use int8')
    return parser


def eval_reader(data_dir, batch_size, crop_size, resize_size):
    val_reader = ImageNetDataset(
        mode='val',
        data_dir=data_dir,
        crop_size=crop_size,
        resize_size=resize_size)
    val_loader = DataLoader(
        val_reader,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=0)
    return val_loader


class Predictor(object):
    def __init__(self, config):

        # HALF precission predict only work when using tensorrt
        if config['use_fp16'] is True:
            assert config['use_tensorrt'] is True
        self.config = config

        self.paddle_predictor = self.create_paddle_predictor()
        input_names = self.paddle_predictor.get_input_names()
        self.input_tensor = self.paddle_predictor.get_input_handle(input_names[
            0])

        output_names = self.paddle_predictor.get_output_names()
        self.output_tensor = self.paddle_predictor.get_output_handle(
            output_names[0])

    def create_paddle_predictor(self):
        inference_model_dir = self.config['model_dir']
        model_file = os.path.join(inference_model_dir,
                                  self.config['model_filename'])
        params_file = os.path.join(inference_model_dir,
                                   self.config['params_filename'])
        config = paddle.inference.Config(model_file, params_file)
        precision = paddle.inference.Config.Precision.Float32
        if self.config['use_int8']:
            precision = paddle.inference.Config.Precision.Int8
        elif self.config['use_fp16']:
            precision = paddle.inference.Config.Precision.Half

        if self.config['use_gpu']:
            config.enable_use_gpu(self.config['gpu_mem'], 0)
        else:
            config.disable_gpu()
            if self.config['enable_mkldnn']:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
        config.set_cpu_math_library_num_threads(self.config['cpu_num_threads'])

        if self.config['enable_profile']:
            config.enable_profile()
        config.switch_ir_optim(self.config['ir_optim'])  # default true
        if self.config['use_tensorrt']:
            config.enable_tensorrt_engine(
                precision_mode=precision,
                max_batch_size=self.config['batch_size'],
                workspace_size=1 << 30,
                min_subgraph_size=30,
                use_calib_mode=False)

        config.enable_memory_optim()
        # use zero copy
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)

        return predictor

    def predict(self):
        test_num = 1000
        test_time = 0.0
        for i in range(0, test_num + 10):
            inputs = np.random.rand(self.config['batch_size'], 3,
                                    self.config['img_size'],
                                    self.config['img_size']).astype(np.float32)
            start_time = time.time()
            self.input_tensor.copy_from_cpu(inputs)
            self.paddle_predictor.run()
            batch_output = self.output_tensor.copy_to_cpu().flatten()
            if i >= 10:
                test_time += time.time() - start_time
            time.sleep(0.01)  # sleep for T4 GPU

        fp_message = "FP16" if self.config['use_fp16'] else "FP32"
        fp_message = "INT8" if self.config['use_int8'] else fp_message
        trt_msg = "using tensorrt" if self.config[
            'use_tensorrt'] else "not using tensorrt"
        print("{0}\t{1}\tbatch size: {2}\ttime(ms): {3}".format(
            trt_msg, fp_message, config[
                'batch_size'], 1000 * test_time / test_num))

    def eval(self):
        img_size = self.config.get("img_size", 224)
        resize_size = self.config.get("resize_size", 256)
        val_loader = eval_reader(
            self.config['data_dir'],
            batch_size=self.config['batch_size'],
            crop_size=img_size,
            resize_size=resize_size)
        results = []

        with tqdm(
                total=len(val_loader),
                bar_format='Evaluation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
                ncols=80) as t:
            for batch_id, (image, label) in enumerate(val_loader):
                use_onnx = self.config.get("use_onnx", False)
                if not use_onnx:
                    input_names = self.paddle_predictor.get_input_names()
                    input_tensor = self.paddle_predictor.get_input_handle(
                        input_names[0])
                    output_names = self.paddle_predictor.get_output_names()
                    output_tensor = self.paddle_predictor.get_output_handle(
                        output_names[0])
                else:
                    input_names = self.paddle_predictor.get_inputs()[0].name
                    output_names = self.paddle_predictor.get_outputs()[0].name
                image = np.array(image)
                if not use_onnx:
                    input_tensor.copy_from_cpu(image)
                    self.paddle_predictor.run()
                    batch_output = output_tensor.copy_to_cpu()
                else:
                    batch_output = self.paddle_predictor.run(
                        output_names=[output_names],
                        input_feed={input_names: image})[0]
                label = np.array(label)
                sort_array = batch_output.argsort(axis=1)
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
        print('Evaluation result: top1: {}, top5: {}'.format(result[0], result[
            1]))


if __name__ == "__main__":
    parser = argsparser()
    args = parser.parse_args()
    global config
    config = load_config(args.config_path)
    if args.model_dir != config['model_dir']:
        config['model_dir'] = args.model_dir
    if args.use_fp16 != config['use_fp16']:
        config['use_fp16'] = args.use_fp16
    if args.use_int8 != config['use_int8']:
        config['use_int8'] = args.use_int8
    predictor = Predictor(config)
    predictor.predict()
    predictor.eval()
