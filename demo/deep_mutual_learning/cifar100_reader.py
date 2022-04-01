# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
# for python2/python3 compatiablity
try:
    import cPickle
except:
    import _pickle as cPickle

IMAGE_SIZE = 32
IMAGE_DEPTH = 3
CIFAR_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
CIFAR_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

URL_PREFIX = 'https://www.cs.toronto.edu/~kriz/'
CIFAR100_URL = URL_PREFIX + 'cifar-100-python.tar.gz'
CIFAR100_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'
paddle.dataset.common.DATA_HOME = "dataset/"


def preprocess(sample, is_training):
    image_array = sample.reshape(IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE)
    rgb_array = np.transpose(image_array, (1, 2, 0))
    img = Image.fromarray(rgb_array, 'RGB')

    if is_training:
        # pad, ramdom crop, random_flip_left_right, random_rotation
        img = ImageOps.expand(img, (4, 4, 4, 4), fill=0)
        left_top = np.random.randint(8, size=2)
        img = img.crop((left_top[1], left_top[0], left_top[1] + IMAGE_SIZE,
                        left_top[0] + IMAGE_SIZE))
        if np.random.randint(2):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        random_angle = np.random.randint(-15, 15)
        img = img.rotate(random_angle, Image.NEAREST)
    img = np.array(img).astype(np.float32)

    img_float = img / 255.0
    img = (img_float - CIFAR_MEAN) / CIFAR_STD

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


def cifar100_reader(file_name, data_name, is_shuffle):
    with tarfile.open(file_name, mode='r') as f:
        names = [
            each_item.name for each_item in f if data_name in each_item.name
        ]
        names.sort()
        datasets = []
        for name in names:
            print("Reading file " + name)
            try:
                batch = cPickle.load(f.extractfile(name), encoding='iso-8859-1')
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
    name = 'train' if is_train else 'test'
    datasets = cifar100_reader(
        paddle.dataset.common.download(CIFAR100_URL, 'cifar', CIFAR100_MD5),
        name, is_shuffle)
    reader = reader_generator(datasets, batch_size, is_train, is_shuffle)
    return reader
