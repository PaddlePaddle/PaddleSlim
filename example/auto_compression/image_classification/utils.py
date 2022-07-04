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

import os
import argparse
import base64
import shutil
import cv2
import numpy as np


def preprocess(img, args):
    resize_op = ResizeImage(resize_short=args.resize_short)
    img = resize_op(img)
    crop_op = CropImage(size=(args.resize, args.resize))
    img = crop_op(img)
    if args.normalize:
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        img_scale = 1.0 / 255.0
        normalize_op = NormalizeImage(
            scale=img_scale, mean=img_mean, std=img_std)
        img = normalize_op(img)
    tensor_op = ToTensor()
    img = tensor_op(img)
    return img


def postprocess(batch_outputs, topk=5, multilabel=False):
    batch_results = []
    for probs in batch_outputs:
        results = []
        if multilabel:
            index = np.where(probs >= 0.5)[0].astype('int32')
        else:
            index = probs.argsort(axis=0)[-topk:][::-1].astype("int32")
        clas_id_list = []
        score_list = []
        for i in index:
            clas_id_list.append(i.item())
            score_list.append(probs[i].item())
        batch_results.append({"clas_ids": clas_id_list, "scores": score_list})
    return batch_results


class ResizeImage(object):
    def __init__(self, resize_short=None):
        self.resize_short = resize_short

    def __call__(self, img):
        img_h, img_w = img.shape[:2]
        percent = float(self.resize_short) / min(img_w, img_h)
        w = int(round(img_w * percent))
        h = int(round(img_h * percent))
        return cv2.resize(img, (w, h))


class CropImage(object):
    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]
        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        return img[h_start:h_end, w_start:w_end, :]


class NormalizeImage(object):
    def __init__(self, scale=None, mean=None, std=None):
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        return (img.astype('float32') * self.scale - self.mean) / self.std


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = img.transpose((2, 0, 1))
        return img
