# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import random
import functools
import numpy as np
import paddle
from PIL import Image, ImageEnhance
from paddle.io import Dataset

random.seed(0)
np.random.seed(0)


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    return img


def crop_image(img, target_size, center):
    width, height = img.size
    if center == True:
        w_start = (width - target_size[-2]) / 2
        h_start = (height - target_size[1]) / 2
    else:
        w_start = np.random.randint(0, width - target_size[-2] + 1)
        h_start = np.random.randint(0, height - target_size[-1] + 1)
    w_end = w_start + target_size[-2]
    h_end = h_start + target_size[-1]
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def random_crop(img, image_shape, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.size[0]) / img.size[1]) / (w**2),
                (float(img.size[1]) / img.size[0]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.size[0] * img.size[1] * np.random.uniform(scale_min,
                                                                scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = np.random.randint(0, img.size[0] - w + 1)
    j = np.random.randint(0, img.size[1] - h + 1)

    img = img.crop((i, j, i + w, j + h))
    img = img.resize((image_shape[-2], image_shape[-1]), Image.LANCZOS)
    return img


def rotate_image(img):
    angle = np.random.randint(-10, 11)
    img = img.rotate(angle)
    return img


def distort_color(img):
    def random_brightness(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)

    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)

    return img


def process_image(sample, mode, image_shape, color_jitter, rotate):
    img_path = sample[0]

    try:
        img = Image.open(img_path)
    except:
        print(img_path, "not exists!")
        return None
    if mode == 'train':
        if rotate: img = rotate_image(img)
        img = random_crop(img, image_shape)
    else:
        img = resize_short(img, target_size=256)
        img = crop_image(img, target_size=image_shape, center=True)
    if mode == 'train':
        if color_jitter:
            img = distort_color(img)
        if np.random.randint(0, 2) == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    if mode == 'train' or mode == 'val':
        return img, sample[1]
    elif mode == 'test':
        return [img]


class ImageNetDataset(Dataset):
    def __init__(self, data_dir, image_shape=[3, 224, 224], mode='train'):
        super(ImageNetDataset, self).__init__()
        self.data_dir = data_dir
        self.image_shape = image_shape
        train_file_list = os.path.join(data_dir, 'train_list.txt')
        val_file_list = os.path.join(data_dir, 'val_list.txt')
        self.mode = mode
        if mode == 'train' or mode == 'infer_train':
            with open(train_file_list) as flist:
                full_lines = [line.strip() for line in flist]
                np.random.shuffle(full_lines)
                lines = full_lines
            self.data = [line.split() for line in lines]
        else:
            with open(val_file_list) as flist:
                lines = [line.strip() for line in flist]
                self.data = [line.split() for line in lines]

    def __getitem__(self, index):
        sample = self.data[index]
        data_path = os.path.join(self.data_dir, sample[0])
        if self.mode == 'train':
            data, label = process_image(
                [data_path, sample[1]],
                mode='train',
                image_shape=self.image_shape,
                color_jitter=False,
                rotate=False)
            return data, np.array([label]).astype('int64')
        elif self.mode == 'val':
            data, label = process_image(
                [data_path, sample[1]],
                mode='val',
                image_shape=self.image_shape,
                color_jitter=False,
                rotate=False)
            return data, np.array([label]).astype('int64')
        elif self.mode == 'test' or self.mode == 'infer_train':
            data = process_image(
                [data_path, sample[1]],
                mode='test',
                image_shape=self.image_shape,
                color_jitter=False,
                rotate=False)
            return data

    def __len__(self):
        return len(self.data)


def train(data_dir,
          image_shape=[3, 224, 224],
          batch_size=1,
          shuffle=True,
          num_workers=1):
    train_dataset = ImageNetDataset(data_dir, image_shape, 'train')
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        return_list=True,
        num_workers=num_workers)
    return train_loader


def val(data_dir,
        image_shape=[3, 224, 224],
        batch_size=1,
        shuffle=False,
        num_workers=1):
    val_dataset = ImageNetDataset(data_dir, image_shape, 'val')
    valid_loader = paddle.io.DataLoader(
        val_dataset,
        # places=place,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        return_list=True,
        num_workers=num_workers)
    return valid_loader


def test(data_dir,
         image_shape=[3, 224, 224],
         batch_size=1,
         shuffle=False,
         num_workers=1):
    test_dataset = ImageNetDataset(data_dir, image_shape, 'test')
    test_loader = paddle.io.DataLoader(
        test_dataset,
        # places=place,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        return_list=True,
        num_workers=num_workers)
    return test_loader


def static_train(data_dir,
                 image_shape=[3, 224, 224],
                 batch_size=1,
                 shuffle=True,
                 num_workers=1):
    train_dataset = ImageNetDataset(data_dir, image_shape, 'train')
    image_shape = [3, 224, 224]
    image = paddle.static.data(
        name='inputs', shape=[None] + image_shape, dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    train_loader = paddle.io.DataLoader(
        train_dataset,
        feed_list=[image, label],
        drop_last=False,
        return_list=False,
        batch_size=64,
        shuffle=False)
    return train_loader


def static_train(data_dir,
                 image_shape=[3, 224, 224],
                 batch_size=1,
                 shuffle=True,
                 num_workers=1):
    val_dataset = ImageNetDataset(data_dir, image_shape, 'val')
    image_shape = [3, 224, 224]
    image = paddle.static.data(
        name='inputs', shape=[None] + image_shape, dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    val_loader = paddle.io.DataLoader(
        val_dataset,
        feed_list=[image, label],
        drop_last=False,
        return_list=False,
        batch_size=64,
        shuffle=False)
    return val_loader


def static_train(data_dir,
                 image_shape=[3, 224, 224],
                 batch_size=1,
                 shuffle=True,
                 num_workers=1):
    test_dataset = ImageNetDataset(data_dir, image_shape, 'test')
    image_shape = [3, 224, 224]
    image = paddle.static.data(
        name='inputs', shape=[None] + image_shape, dtype='float32')
    test_loader = paddle.io.DataLoader(
        test_dataset,
        feed_list=[image],
        drop_last=False,
        return_list=False,
        batch_size=64,
        shuffle=False)
    return test_loader
