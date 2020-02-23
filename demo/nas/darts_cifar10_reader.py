# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
from PIL import ImageOps
import os
import math
import random
import tarfile
import functools
import numpy as np
from PIL import Image, ImageEnhance
import paddle
try:
    import cPickle
except:
    import _pickle as cPickle

IMAGE_SIZE = 32
IMAGE_DEPTH = 3
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

URL_PREFIX = 'https://www.cs.toronto.edu/~kriz/'
CIFAR10_URL = URL_PREFIX + 'cifar-10-python.tar.gz'
CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'

paddle.dataset.common.DATA_HOME = "dataset/"

THREAD = 16
BUF_SIZE = 10240

num_workers = 4
use_multiprocess = True
cutout = True
cutout_length = 16


def preprocess(sample, is_training):
    image_array = sample.reshape(IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE)
    rgb_array = np.transpose(image_array, (1, 2, 0))
    img = Image.fromarray(rgb_array, 'RGB')

    if is_training:
        # pad, ramdom crop, random_flip_left_right
        img = ImageOps.expand(img, (4, 4, 4, 4), fill=0)
        left_top = np.random.randint(8, size=2)
        img = img.crop((left_top[1], left_top[0], left_top[1] + IMAGE_SIZE,
                        left_top[0] + IMAGE_SIZE))
        if np.random.randint(2):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = np.array(img).astype(np.float32)

    img_float = img / 255.0
    img = (img_float - CIFAR_MEAN) / CIFAR_STD

    if is_training and cutout:
        center = np.random.randint(IMAGE_SIZE, size=2)
        offset_width = max(0, center[0] - cutout_length // 2)
        offset_height = max(0, center[1] - cutout_length // 2)
        target_width = min(center[0] + cutout_length // 2, IMAGE_SIZE)
        target_height = min(center[1] + cutout_length // 2, IMAGE_SIZE)

        for i in range(offset_height, target_height):
            for j in range(offset_width, target_width):
                img[i][j][:] = 0.0

    img = np.transpose(img, (2, 0, 1))
    return img


def reader_generator(datasets, batch_size, is_training, is_shuffle):
    def read_batch(datasets):
        if is_shuffle:
            random.shuffle(datasets)
        for im, label in datasets:
            im = preprocess(im, is_training)
            yield im, [int(label)]

    def reader():
        batch_data = []
        batch_label = []
        for data in read_batch(datasets):
            batch_data.append(data[0])
            batch_label.append(data[1])
            if len(batch_data) == batch_size:
                batch_data = np.array(batch_data, dtype='float32')
                batch_label = np.array(batch_label, dtype='int64')
                batch_out = [batch_data, batch_label]
                yield batch_out
                batch_data = []
                batch_label = []

    return reader


def cifar10_reader(file_name, data_name, is_shuffle):
    with tarfile.open(file_name, mode='r') as f:
        names = [
            each_item.name for each_item in f if data_name in each_item.name
        ]
        names.sort()
        datasets = []
        for name in names:
            print("Reading file " + name)
            try:
                batch = cPickle.load(
                    f.extractfile(name), encoding='iso-8859-1')
            except:
                batch = cPickle.load(f.extractfile(name))
            data = batch['data']
            labels = batch.get('labels', batch.get('fine_labels', None))
            assert labels is not None
            dataset = zip(data, labels)
            datasets.extend(dataset)
        if is_shuffle:
            random.shuffle(datasets)
    return datasets


def train_valid(batch_size, is_train, is_shuffle):
    name = 'data_batch' if is_train else 'test_batch'
    datasets = cifar10_reader(
        paddle.dataset.common.download(CIFAR10_URL, 'cifar', CIFAR10_MD5),
        name, is_shuffle)
    n = int(math.ceil(len(datasets) //
                      num_workers)) if use_multiprocess else len(datasets)
    datasets_lists = [datasets[i:i + n] for i in range(0, len(datasets), n)]
    multi_readers = []
    for pid in range(len(datasets_lists)):
        multi_readers.append(
            reader_generator(datasets_lists[pid], batch_size, is_train,
                             is_shuffle))
    if use_multiprocess:
        reader = paddle.reader.multiprocess_reader(multi_readers, False)
    else:
        reader = multi_readers[0]
    return reader
