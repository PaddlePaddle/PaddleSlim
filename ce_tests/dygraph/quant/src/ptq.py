# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import argparse
import contextlib
import os

import time
import math
import numpy as np

import paddle
import paddle.vision.models as models

from imagenet_dataset import ImageNetDataset
from paddleslim import PTQ


def calibrate(model, dataset, batch_num, batch_size):
    data_loader = paddle.io.DataLoader(
        dataset, batch_size=batch_size, num_workers=5)

    acc_list = []
    for idx, data in enumerate(data_loader()):
        img = data[0]
        label = data[1]

        out = model(img)

        acc = paddle.metric.accuracy(out, label)
        acc_list.append(acc.numpy())

        if (idx + 1) % 50 == 0:
            print("idx:" + str(idx))
        if (batch_num > 0) and (idx + 1 >= batch_num):
            break

    return np.mean(acc_list)


def main():
    # 1 load model
    model_list = [x for x in models.__dict__["__all__"]]
    assert FLAGS.arch in model_list, "Expected FLAGS.arch in {}, but received {}".format(
        model_list, FLAGS.arch)
    fp32_model = models.__dict__[FLAGS.arch](pretrained=True)
    fp32_model.eval()

    val_dataset = ImageNetDataset(FLAGS.data, mode='val')

    # 2 quantizations
    ptq = PTQ()
    quant_model = ptq.quantize(fp32_model)

    print("Calibrate")
    calibrate(quant_model, val_dataset, FLAGS.quant_batch_num,
              FLAGS.quant_batch_size)

    # 3 save
    quant_output_dir = os.path.join(FLAGS.output_dir, FLAGS.arch, "int8_infer",
                                    "model")
    input_spec = paddle.static.InputSpec(
        shape=[None, 3, 224, 224], dtype='float32')
    ptq.save_quantized_model(quant_model, quant_output_dir, [input_spec])

    fp32_output_dir = os.path.join(FLAGS.output_dir, FLAGS.arch, "fp32_infer",
                                   "model")
    paddle.jit.save(fp32_model, fp32_output_dir, [input_spec])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Quantization on ImageNet")

    # model
    parser.add_argument(
        "--arch", type=str, default='mobilenet_v2', help="model name")
    parser.add_argument(
        "--output_dir", type=str, default='output', help="save dir")

    # data
    parser.add_argument(
        '--data',
        default="/dataset/ILSVRC2012",
        help='path to dataset (should have subdirectories named "train" and "val"'
    )
    # train
    parser.add_argument(
        "--quant_batch_num", default=10, type=int, help="batch num for quant")
    parser.add_argument(
        "--quant_batch_size", default=10, type=int, help="batch size for quant")

    FLAGS = parser.parse_args()
    assert FLAGS.data, "error: must provide data path"

    main()
