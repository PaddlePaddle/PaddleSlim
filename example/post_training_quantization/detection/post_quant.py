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
from ppdet.core.workspace import load_config, merge_config
from ppdet.core.workspace import create
from paddleslim.quant import quant_post_static


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='ptq_out',
        help="directory to save compressed model.")
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")
    parser.add_argument(
        '--algo', type=str, default='KL', help="post quant algo.")

    return parser


def reader_wrapper(reader, input_list):
    def gen():
        for data in reader:
            in_dict = {}
            if isinstance(input_list, list):
                for input_name in input_list:
                    in_dict[input_name] = data[input_name]
            elif isinstance(input_list, dict):
                for input_name in input_list.keys():
                    in_dict[input_list[input_name]] = data[input_name]
            yield in_dict

    return gen


def main():
    global config
    config = load_config(FLAGS.config_path)

    train_loader = create('EvalReader')(config['TrainDataset'],
                                        config['worker_num'],
                                        return_list=True)
    train_loader = reader_wrapper(train_loader, config['input_list'])

    place = paddle.CUDAPlace(0) if FLAGS.devices == 'gpu' else paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    quant_post_static(
        executor=exe,
        model_dir=config["model_dir"],
        quantize_model_path=FLAGS.save_dir,
        data_loader=train_loader,
        model_filename=config["model_filename"],
        params_filename=config["params_filename"],
        batch_size=4,
        batch_nums=64,
        algo=FLAGS.algo,
        hist_percent=0.999,
        is_full_quantize=False,
        bias_correction=False,
        onnx_format=False,
        skip_tensor_list=config['skip_tensor_list']
        if 'skip_tensor_list' in config else None)


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()

    assert FLAGS.devices in ['cpu', 'gpu', 'xpu', 'npu']
    paddle.set_device(FLAGS.devices)

    main()
