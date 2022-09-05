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
from imagenet_reader import ImageNetDataset, process_image


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./MobileNetV1_infer',
        help='model directory')
    parser.add_argument(
        '--model_filename',
        type=str,
        default='inference.pdmodel',
        help='model file name')
    parser.add_argument(
        '--params_filename',
        type=str,
        default='inference.pdiparams',
        help='params file name')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--resize_size', type=int, default=256)
    parser.add_argument(
        '--eval', type=bool, default=False, help='Whether to evaluate')
    parser.add_argument('--data_path', type=str, default='./ILSVRC2012/')
    parser.add_argument(
        '--use_gpu', type=bool, default=False, help='Whether to use gpu')
    parser.add_argument(
        '--enable_mkldnn',
        type=bool,
        default=False,
        help='Whether to use mkldnn')
    parser.add_argument(
        '--cpu_num_threads', type=int, default=10, help='Number of cpu threads')
    parser.add_argument(
        '--use_fp16', type=bool, default=False, help='Whether to use fp16')
    parser.add_argument(
        '--use_int8', type=bool, default=False, help='Whether to use int8')
    parser.add_argument(
        '--use_tensorrt',
        type=bool,
        default=True,
        help='Whether to use tensorrt')
    parser.add_argument(
        '--enable_profile',
        type=bool,
        default=False,
        help='Whether to enable profile')
    parser.add_argument('--gpu_mem', type=int, default=8000, help='GPU memory')
    parser.add_argument('--ir_optim', type=bool, default=True)
    return parser


def eval_reader(data_dir, batch_size, crop_size, resize_size):
    val_reader = ImageNetDataset(
        mode='val',
        data_dir=data_dir,
        crop_size=crop_size,
        resize_size=resize_size)
    val_loader = DataLoader(
        val_reader,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)
    return val_loader


class Predictor(object):
    def __init__(self, args):

        # HALF precission predict only work when using tensorrt
        if args.use_fp16 is True:
            assert args.use_tensorrt is True
        self.args = args

        self.paddle_predictor = self.create_paddle_predictor()
        input_names = self.paddle_predictor.get_input_names()
        self.input_tensor = self.paddle_predictor.get_input_handle(input_names[
            0])

        output_names = self.paddle_predictor.get_output_names()
        self.output_tensor = self.paddle_predictor.get_output_handle(
            output_names[0])

    def create_paddle_predictor(self):
        inference_model_dir = self.args.model_dir
        model_file = os.path.join(inference_model_dir, self.args.model_filename)
        params_file = os.path.join(inference_model_dir,
                                   self.args.params_filename)
        config = paddle.inference.Config(model_file, params_file)
        precision = paddle.inference.Config.Precision.Float32
        if self.args.use_int8:
            precision = paddle.inference.Config.Precision.Int8
        elif self.args.use_fp16:
            precision = paddle.inference.Config.Precision.Half

        if self.args.use_gpu:
            config.enable_use_gpu(self.args.gpu_mem, 0)
        else:
            config.disable_gpu()
            if self.args.enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
        config.set_cpu_math_library_num_threads(self.args.cpu_num_threads)

        if self.args.enable_profile:
            config.enable_profile()
        config.switch_ir_optim(self.args.ir_optim)  # default true
        if self.args.use_tensorrt:
            config.enable_tensorrt_engine(
                precision_mode=precision,
                max_batch_size=self.args.batch_size,
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
            inputs = np.random.rand(self.args.batch_size, 3, self.args.img_size,
                                    self.args.img_size).astype(np.float32)
            start_time = time.time()
            self.input_tensor.copy_from_cpu(inputs)
            self.paddle_predictor.run()
            batch_output = self.output_tensor.copy_to_cpu().flatten()
            if i >= 10:
                test_time += time.time() - start_time
            time.sleep(0.01)  # sleep for T4 GPU

        fp_message = "FP16" if self.args.use_fp16 else "FP32"
        fp_message = "INT8" if self.args.use_int8 else fp_message
        trt_msg = "using tensorrt" if self.args.use_tensorrt else "not using tensorrt"
        print("{0}\t{1}\tbatch size: {2}\ttime(ms): {3}".format(
            trt_msg, fp_message, args.batch_size, 1000 * test_time / test_num))

    def eval(self):
        if os.path.exists(self.args.data_path):
            val_loader = eval_reader(
                self.args.data_path,
                batch_size=self.args.batch_size,
                crop_size=self.args.img_size,
                resize_size=self.args.resize_size)
        else:
            image = np.ones((1, 3, self.args.img_size,
                             self.args.img_size)).astype(np.float32)
            label = None
            val_loader = [[image, label]]
        results = []
        with tqdm(
                total=len(val_loader),
                bar_format='Evaluation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
                ncols=80) as t:
            for batch_id, (image, label) in enumerate(val_loader):
                input_names = self.paddle_predictor.get_input_names()
                input_tensor = self.paddle_predictor.get_input_handle(
                    input_names[0])
                output_names = self.paddle_predictor.get_output_names()
                output_tensor = self.paddle_predictor.get_output_handle(
                    output_names[0])

                image = np.array(image)

                input_tensor.copy_from_cpu(image)
                self.paddle_predictor.run()
                batch_output = output_tensor.copy_to_cpu()
                sort_array = batch_output.argsort(axis=1)
                top_1_pred = sort_array[:, -1:][:, ::-1]
                if label is None:
                    results.append(top_1_pred)
                    break
                label = np.array(label)
                top_1 = np.mean(label == top_1_pred)
                top_5_pred = sort_array[:, -5:][:, ::-1]
                acc_num = 0
                for i in range(len(label)):
                    if label[i][0] in top_5_pred[i]:
                        acc_num += 1
                top_5 = float(acc_num) / len(label)
                results.append([top_1, top_5])

            result = np.mean(np.array(results), axis=0)
        print('Evaluation result: {}'.format(result[0]))


if __name__ == "__main__":
    parser = argsparser()
    global args
    args = parser.parse_args()
    predictor = Predictor(args)
    predictor.predict()
    if args.eval:
        predictor.eval()
