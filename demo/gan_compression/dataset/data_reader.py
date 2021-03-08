#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import print_function
from six.moves import range
from PIL import Image, ImageOps

import gzip
import numpy as np
import argparse
import struct
import os
import paddle
import random


def RandomCrop(img, crop_w, crop_h):
    w, h = img.size[0], img.size[1]
    i = np.random.randint(0, w - crop_w)
    j = np.random.randint(0, h - crop_h)
    return img.crop((i, j, i + crop_w, j + crop_h))


def CentorCrop(img, crop_w, crop_h):
    w, h = img.size[0], img.size[1]
    i = int((w - crop_w) / 2.0)
    j = int((h - crop_h) / 2.0)
    return img.crop((i, j, i + crop_w, j + crop_h))


def RandomHorizonFlip(img):
    i = np.random.rand()
    if i > 0.5:
        img = ImageOps.mirror(img)
    return img


class ReaderCreator:
    def __init__(self, *args, **kwcfgs):
        raise NotImplementedError

    def make_reader(self, cfgs):
        raise NotImplementedError


class SingleDataReader(ReaderCreator):
    def __init__(self, list_filename, cfgs, mode='TEST'):
        self.cfgs = cfgs
        self.mode = mode
        with open(list_filename) as f:
            self.lines = f.readlines()
        (self.data_dir, _) = os.path.split(list_filename)
        self.id2name = {}
        if self.mode == "TRAIN":
            self.shuffle = self.cfgs.shuffle
        else:
            self.shuffle = False

    def len(self):
        return self.len(self.lines)

    def make_reader(self):
        def reader():
            batch_out_1 = []
            batch_out_name = []
            if self.shuffle:
                np.random.shuffle(self.lines)

            for i, fileA in enumerate(self.lines):
                fileA = fileA.strip('\n\r\t').split(' ')[0]
                self.id2name[i] = os.path.basename(fileA)
                imgA = Image.open(os.path.join(self.data_dir, fileA)).convert(
                    'RGB')

                if self.mode == 'TRAIN':
                    imgA = imgA.resize((self.cfgs.image_size,
                                        self.cfgs.image_size), Image.BICUBIC)

                    if self.cfgs.crop_type == 'Centor':
                        imgA = CentorCrop(imgA, self.cfgs.crop_size,
                                          self.cfgs.crop_size)
                    elif self.cfgs.crop_type == 'Random':
                        imgA = RandomCrop(imgA, self.cfgs.crop_size,
                                          self.cfgs.crop_size)
                    if self.cfgs.flip:
                        imgA = RandomHorizonFlip(imgA)
                else:
                    imgA = imgA.resize((self.cfgs.crop_size,
                                        self.cfgs.crop_size), Image.BICUBIC)

                imgA = (np.array(imgA).astype('float32') / 255.0 - 0.5) / 0.5
                imgA = imgA.transpose([2, 0, 1])

                batch_out_1.append(imgA)
                batch_out_name.append(i)

                if len(batch_out_1) == self.cfgs.batch_size:
                    yield batch_out_1, batch_out_name
                    batch_out_1 = []
                    batch_out_name = []

        return reader


class PairDataReader(ReaderCreator):
    def __init__(self, list_filename_A, list_filename_B, cfgs, mode='TRAIN'):
        self.cfgs = cfgs
        self.mode = mode
        with open(list_filename_A) as f:
            self.lines_A = f.readlines()
        with open(list_filename_B) as f:
            self.lines_B = f.readlines()
        (self.data_dir, _) = os.path.split(list_filename_A)
        self._max_dataset_size = max(len(self.lines_A), len(self.lines_B))
        self.id2name = {}
        if self.mode == "TRAIN":
            if len(self.lines_A) < self._max_dataset_size:
                rundant = self._max_dataset_size % len(self.lines_A)
                self.lines_A.extend(self.lines_A[:rundant])
            if len(self.lines_B) < self._max_dataset_size:
                rundant = self._max_dataset_size % len(self.lines_B)
                self.lines_B.extend(self.lines_B[:rundant])
            self.shuffle = self.cfgs.shuffle
        else:
            self.shuffle = False

    def len(self):
        return self._max_dataset_size

    def make_reader(self):
        def reader():
            batch_out_1 = []
            batch_out_2 = []
            batch_out_name = []

            if self.shuffle:
                np.random.shuffle(self.lines_B)

            for i, (fileA,
                    fileB) in enumerate(zip(self.lines_A, self.lines_B)):
                fileA = fileA.strip('\n\r\t').split(' ')[0]
                fileB = fileB.strip('\n\r\t').split(' ')[0]
                self.id2name[i] = os.path.basename(
                    fileA) + '###' + os.path.basename(fileB)

                imgA = Image.open(os.path.join(self.data_dir, fileA)).convert(
                    'RGB')
                imgB = Image.open(os.path.join(self.data_dir, fileB)).convert(
                    'RGB')

                if self.mode == 'TRAIN':
                    imgA = imgA.resize((self.cfgs.image_size,
                                        self.cfgs.image_size), Image.BICUBIC)
                    imgB = imgB.resize((self.cfgs.image_size,
                                        self.cfgs.image_size), Image.BICUBIC)

                    if self.cfgs.crop_type == 'Centor':
                        imgA = CentorCrop(imgA, self.cfgs.crop_size,
                                          self.cfgs.crop_size)
                        imgB = CentorCrop(imgB, self.cfgs.crop_size,
                                          self.cfgs.crop_size)
                    elif self.cfgs.crop_type == 'Random':
                        imgA = RandomCrop(imgA, self.cfgs.crop_size,
                                          self.cfgs.crop_size)
                        imgB = RandomCrop(imgB, self.cfgs.crop_size,
                                          self.cfgs.crop_size)
                    if self.cfgs.flip:
                        imgA = RandomHorizonFlip(imgA)
                        imgB = RandomHorizonFlip(imgB)
                else:
                    imgA = imgA.resize((self.cfgs.crop_size,
                                        self.cfgs.crop_size), Image.BICUBIC)
                    imgB = imgB.resize((self.cfgs.crop_size,
                                        self.cfgs.crop_size), Image.BICUBIC)

                imgA = (np.array(imgA).astype('float32') / 255.0 - 0.5) / 0.5
                imgA = imgA.transpose([2, 0, 1])

                imgB = (np.array(imgB).astype('float32') / 255.0 - 0.5) / 0.5
                imgB = imgB.transpose([2, 0, 1])

                batch_out_1.append(imgA)
                batch_out_2.append(imgB)
                batch_out_name.append(i)

                if len(batch_out_1) == self.cfgs.batch_size:
                    yield batch_out_1, batch_out_2, batch_out_name
                    batch_out_2 = []
                    batch_out_1 = []
                    batch_out_name = []

        return reader


class DataReader(object):
    def __init__(self, cfgs, mode='TRAIN'):
        self.mode = mode
        self.cfgs = cfgs

    def make_data(self, direction='AtoB'):
        if self.cfgs.model == 'cycle_gan':
            dataset_dir = os.path.join(self.cfgs.dataroot, self.cfgs.dataset)
            fileB_list = None
            if self.mode == 'TRAIN':
                fileA_list = os.path.join(dataset_dir, "trainA.txt")
                fileB_list = os.path.join(dataset_dir, "trainB.txt")
            else:
                if direction == 'AtoB':
                    fileA_list = os.path.join(dataset_dir, "testA.txt")
                else:
                    fileA_list = os.path.join(dataset_dir, "testB.txt")

            if fileB_list is not None:
                train_reader = PairDataReader(
                    list_filename_A=fileA_list,
                    list_filename_B=fileB_list,
                    cfgs=self.cfgs,
                    mode=self.mode)
            else:
                train_reader = SingleDataReader(
                    list_filename=fileA_list, cfgs=self.cfgs, mode=self.mode)

            reader = train_reader.make_reader()
            id2name = train_reader.id2name
            return reader, id2name
