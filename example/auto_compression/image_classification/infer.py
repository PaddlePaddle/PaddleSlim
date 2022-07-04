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
from utils import preprocess, postprocess
import paddle
from paddle.inference import create_predictor
from paddleslim.auto_compression.config_helpers import load_config


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default='configs/infer.yaml',
        help='config file path')
    return parser


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
        inference_model_dir = self.config['inference_model_dir']
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
        config.disable_glog_info()
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
            inputs = np.random.rand(config['batch_size'], 3,
                                    config['image_size'],
                                    config['image_size']).astype(np.float32)
            start_time = time.time()
            self.input_tensor.copy_from_cpu(inputs)
            self.paddle_predictor.run()
            batch_output = self.output_tensor.copy_to_cpu().flatten()
            if i >= 10:
                test_time += time.time() - start_time
            time.sleep(0.01)  # sleep for T4 GPU

        fp_message = "FP16" if config['use_fp16'] else "FP32"
        trt_msg = "using tensorrt" if config[
            'use_tensorrt'] else "not using tensorrt"
        print("{0}\t{1}\tbatch size: {2}\ttime(ms): {3}".format(
            trt_msg, fp_message, config[
                'batch_size'], 1000 * test_time / test_num))


if __name__ == "__main__":
    parser = argsparser()
    args = parser.parse_args()
    config = load_config(args.config_path)
    predictor = Predictor(config)
    predictor.predict()
