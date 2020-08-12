# ================================================================
#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import numpy as np
import six
if six.PY2:
    import scipy.misc as imgreader
else:
    import imageio as imgreader
import os
import paddle
from paddle import fluid


class CASIA_Face(object):
    def __init__(self, root):
        self.root = root

        img_txt_dir = os.path.join(root, 'CASIA-WebFace-112X96.txt')
        image_list = []
        label_list = []
        with open(img_txt_dir) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_dir, label_name = info.split(' ')
            image_list.append(
                os.path.join(root, 'CASIA-WebFace-112X96', image_dir))
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))
        self.shuffle_idx = list(
            np.random.choice(
                len(self.image_list), len(self.image_list), False))

    def reader(self):
        while True:
            if len(self.shuffle_idx) == 0:
                self.shuffle_idx = list(
                    np.random.choice(
                        len(self.image_list), len(self.image_list), False))
                return
            index = self.shuffle_idx.pop()

            img_path = self.image_list[index]
            target = self.label_list[index]

            try:
                img = imgreader.imread(img_path)
            except:
                continue

            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            flip = np.random.choice(2) * 2 - 1
            img = img[:, ::flip, :]
            img = (img - 127.5) / 128.0
            img = img.transpose(2, 0, 1)

            yield img, target

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    data_dir = 'PATH to CASIA dataset'

    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        dataset = CASIA_Face(root=data_dir)
        print(len(dataset))
        print(dataset.class_nums)
        trainloader = paddle.fluid.io.batch(
            dataset.reader, batch_size=1, drop_last=False)
        for i in range(10):
            for data in trainloader():
                img = np.array([x[0] for x in data]).astype('float32')
                img = fluid.dygraph.to_variable(img)
                print(img.shape)
                label = np.array([x[1] for x in data]).astype('int64').reshape(
                    -1, 1)
                label = fluid.dygraph.to_variable(label)
                print(label.shape)
        print(len(dataset))
