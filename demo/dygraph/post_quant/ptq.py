# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import random
import numpy as np
from PIL import Image

import paddle
from paddle.io import Dataset
from paddle.vision.transforms import transforms
import paddle.vision.models as models
import paddle.nn as nn
from paddleslim import PTQ

import sys
sys.path.append(os.path.dirname("__file__"))
sys.path.append(
    os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir))
from models.dygraph.mobilenet_v3 import MobileNetV3_large_x1_0


class ImageNetValDataset(Dataset):
    def __init__(self, data_dir, image_size=224, resize_short_size=256):
        super(ImageNetValDataset, self).__init__()
        val_file_list = os.path.join(data_dir, 'val_list.txt')
        test_file_list = os.path.join(data_dir, 'test_list.txt')
        self.data_dir = data_dir

        normalize = transforms.Normalize(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375])
        self.transform = transforms.Compose([
            transforms.Resize(resize_short_size),
            transforms.CenterCrop(image_size), transforms.Transpose(), normalize
        ])

        with open(val_file_list) as flist:
            lines = [line.strip() for line in flist]
            self.data = [line.split() for line in lines]

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img_path = os.path.join(self.data_dir, img_path)
        img = Image.open(img_path).convert('RGB')
        label = np.array([label]).astype(np.int64)
        return self.transform(img), label

    def __len__(self):
        return len(self.data)


def calibrate(model, dataset, batch_num, batch_size, num_workers=5):
    data_loader = paddle.io.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers)

    for idx, data in enumerate(data_loader()):
        img = data[0]
        label = data[1]

        out = model(img)

        if (idx + 1) % 50 == 0:
            print("idx:" + str(idx))
        if (batch_num > 0) and (idx + 1 >= batch_num):
            break


def main():
    num_workers = 5
    if FLAGS.ce_test:
        # set seed
        seed = 111
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        num_workers = 0

    # 1 load model
    model_list = [x for x in models.__dict__["__all__"]]
    model_list.append('mobilenet_v3')
    assert FLAGS.model in model_list, "Expected FLAGS.model in {}, but received {}".format(
        model_list, FLAGS.model)
    if FLAGS.model == 'mobilenet_v3':
        fp32_model = MobileNetV3_large_x1_0(skip_se_quant=True)
    else:
        fp32_model = models.__dict__[FLAGS.model](pretrained=True)
    if FLAGS.pretrain_weight:
        info_dict = paddle.load(FLAGS.pretrain_weight)
        fp32_model.load_dict(info_dict)
        print('Finish loading model weights:{}'.format(FLAGS.pretrain_weight))
    fp32_model.eval()
    for name, layer in fp32_model.named_sublayers():
        print(name, layer)
    count = 0
    fuse_list = []
    for name, layer in fp32_model.named_sublayers():
        if isinstance(layer, nn.Conv2D):
            fuse_list.append([name])
        if isinstance(layer, nn.BatchNorm2D):
            fuse_list[count].append(name)
            count += 1
    if FLAGS.model == 'resnet50':
        fuse_list = None
    val_dataset = ImageNetValDataset(FLAGS.data)

    # 2 quantizations
    if FLAGS.model == 'mobilenet_v3':
        ptq_config = {
            'activation_quantizer': 'HistQuantizer',
            'upsample_bins': 127,
            'hist_percent': 0.999
        }
        ptq = PTQ(**ptq_config)
    else:
        ptq = PTQ()
    quant_model = ptq.quantize(fp32_model, fuse=FLAGS.fuse, fuse_list=fuse_list)

    print("Start calibrate...")
    calibrate(
        quant_model,
        val_dataset,
        FLAGS.quant_batch_num,
        FLAGS.quant_batch_size,
        num_workers=num_workers)

    # 3 save
    quant_output_dir = os.path.join(FLAGS.output_dir, FLAGS.model, "int8_infer",
                                    "model")
    input_spec = paddle.static.InputSpec(
        shape=[None, 3, 224, 224], dtype='float32')
    ptq.save_quantized_model(quant_model, quant_output_dir, [input_spec])

    fp32_output_dir = os.path.join(FLAGS.output_dir, FLAGS.model, "fp32_infer",
                                   "model")
    paddle.jit.save(fp32_model, fp32_output_dir, [input_spec])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Quantization on ImageNet")

    # model
    parser.add_argument(
        "--model", type=str, default='mobilenet_v3', help="model name")
    parser.add_argument(
        "--pretrain_weight",
        type=str,
        default=None,
        help="pretrain weight path")
    parser.add_argument(
        "--output_dir", type=str, default='output', help="save dir")
    parser.add_argument("--fuse", type=bool, default=False, help="fuse layers")

    # data
    parser.add_argument(
        '--data',
        default="/dataset/ILSVRC2012",
        help='path to dataset (should have subdirectories named "train" and "val"'
    )
    parser.add_argument(
        '--val_dir',
        default="val",
        help='the dir that saves val images for paddle.Model')

    # train
    parser.add_argument(
        "--quant_batch_num", default=10, type=int, help="batch num for quant")
    parser.add_argument(
        "--quant_batch_size", default=10, type=int, help="batch size for quant")
    parser.add_argument(
        '--ce_test', default=False, type=bool, help="Whether to CE test.")

    FLAGS = parser.parse_args()
    assert FLAGS.data, "error: must provide data path"

    main()
