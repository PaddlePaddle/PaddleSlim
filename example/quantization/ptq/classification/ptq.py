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
import six
from inspect import isfunction
import os
import time
import random
from types import FunctionType
from typing import Dict
import numpy as np
from PIL import Image

import paddle
from paddle.io import Dataset
from paddle.vision.transforms import transforms
import paddle.vision.models as models
from paddle.quantization import QuantConfig
from paddle.quantization import PTQ
from tqdm import tqdm
from paddleslim.quant.observers import HistObserver, KLObserver, EMDObserver, MSEObserver, AVGObserver
from paddleslim.quant.observers import MSEChannelWiseWeightObserver, AbsMaxChannelWiseWeightObserver

import sys
sys.path.append(os.path.dirname("__file__"))
sys.path.append(
    os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir))

SUPPORT_MODELS: Dict[str, FunctionType] = {}
for _name, _module in models.__dict__.items():
    if isfunction(_module) and 'pretrained' in _module.__code__.co_varnames:
        SUPPORT_MODELS[_name] = _module

ACTIVATION_OBSERVERS: Dict[str, type] = {
    'hist': HistObserver,
    'kl': KLObserver,
    'emd': EMDObserver,
    'mse': MSEObserver,
    'avg': AVGObserver,
}

WEIGHT_OBSERVERS: Dict[str, type] = {
    'mse_channel_wise': MSEChannelWiseWeightObserver,
    'abs_max_channel_wise': AbsMaxChannelWiseWeightObserver,
}


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
            transforms.CenterCrop(image_size),
            transforms.Transpose(), normalize
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


def test(net, dataset):
    valid_loader = paddle.io.DataLoader(dataset, batch_size=1)
    net.eval()
    batch_id = 0
    acc_top1_ns = []
    acc_top5_ns = []

    eval_reader_cost = 0.0
    eval_run_cost = 0.0
    total_samples = 0
    reader_start = time.time()
    for data in tqdm(valid_loader()):
        eval_reader_cost += time.time() - reader_start
        image = data[0]
        label = data[1]
        eval_start = time.time()

        out = net(image)
        acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
        acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

        eval_run_cost += time.time() - eval_start
        batch_size = image.shape[0]
        total_samples += batch_size

        acc_top1_ns.append(np.mean(acc_top1.numpy()))
        acc_top5_ns.append(np.mean(acc_top5.numpy()))
        batch_id += 1
        reader_start = time.time()
    return np.mean(np.array(acc_top1_ns)), np.mean(np.array(acc_top5_ns))


def calibrate(model, dataset, batch_num, batch_size, num_workers=1):
    data_loader = paddle.io.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers)

    pbar = tqdm(total=batch_num)
    for idx, data in enumerate(data_loader()):
        model(data[0])
        pbar.update(1)
        if (batch_num > 0) and (idx + 1 >= batch_num):
            break
    pbar.close()


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
    fp32_model = SUPPORT_MODELS[FLAGS.model](pretrained=True)
    if FLAGS.pretrain_weight:
        info_dict = paddle.load(FLAGS.pretrain_weight)
        fp32_model.load_dict(info_dict)
        print('Finish loading model weights:{}'.format(FLAGS.pretrain_weight))
    fp32_model.eval()
    val_dataset = ImageNetValDataset(FLAGS.data)

    # 2 quantizations
    activation_observer = ACTIVATION_OBSERVERS[FLAGS.activation_observer]()
    weight_observer = WEIGHT_OBSERVERS[FLAGS.weight_observer]()

    config = QuantConfig(weight=None, activation=None)
    config.add_type_config(
        paddle.nn.Conv2D,
        activation=activation_observer,
        weight=weight_observer)
    ptq = PTQ(config)
    top1, top5 = test(fp32_model, val_dataset)
    print(
        f"\033[31mBaseline(FP32): top1/top5 = {top1*100:.2f}%/{top5*100:.2f}%\033[0m"
    )
    quant_model = ptq.quantize(fp32_model)

    print("Start PTQ calibration for quantization")
    calibrate(
        quant_model,
        val_dataset,
        FLAGS.quant_batch_num,
        FLAGS.quant_batch_size,
        num_workers=num_workers)

    infer_model = ptq.convert(quant_model, inplace=True)

    top1, top5 = test(infer_model, val_dataset)
    print(
        f"\033[31mPTQ with {FLAGS.activation_observer}/{FLAGS.weight_observer}: top1/top5 = {top1*100:.2f}%/{top5*100:.2f}%\033[0m"
    )

    dummy_input = paddle.static.InputSpec(
        shape=[None, 3, 224, 224], dtype='float32')
    paddle.jit.save(infer_model, "./int8_infer", [dummy_input])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Quantization on ImageNet")

    # model
    parser.add_argument(
        "--model",
        type=str,
        choices=SUPPORT_MODELS.keys(),
        default='mobilenet_v1',
        help="model name", )
    parser.add_argument(
        "--pretrain_weight",
        type=str,
        default=None,
        help="pretrain weight path")
    parser.add_argument(
        "--output_dir", type=str, default='output', help="save dir")

    # data
    parser.add_argument(
        '--data',
        default="/dataset/ILSVRC2012",
        help=
        'path to dataset (should have subdirectories named "train" and "val"',
        required=True, )

    parser.add_argument(
        '--val_dir',
        default="val",
        help='the dir that saves val images for paddle.Model')

    # quantization
    parser.add_argument(
        "--activation_observer",
        default='mse',
        type=str,
        choices=ACTIVATION_OBSERVERS.keys(),
        help="batch num for quant")
    parser.add_argument(
        "--weight_observer",
        default='mse_channel_wise',
        choices=WEIGHT_OBSERVERS.keys(),
        type=str,
        help="batch size for quant")

    # train
    parser.add_argument(
        "--quant_batch_num", default=10, type=int, help="batch num for quant")
    parser.add_argument(
        "--quant_batch_size", default=10, type=int, help="batch size for quant")
    parser.add_argument(
        '--ce_test', default=False, type=bool, help="Whether to CE test.")

    FLAGS = parser.parse_args()
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(FLAGS))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")
    main()
