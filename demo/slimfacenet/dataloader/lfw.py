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
import paddle
from paddle import fluid


class LFW(object):
    def __init__(self, imgl, imgr):

        self.imgl_list = imgl
        self.imgr_list = imgr
        self.shuffle_idx = [i for i in range(len(self.imgl_list))]

    def reader(self):
        while True:
            if len(self.shuffle_idx) == 0:
                self.shuffle_idx = [i for i in range(len(self.imgl_list))]
                return
            index = self.shuffle_idx.pop(0)

            imgl = imgreader.imread(self.imgl_list[index])
            if len(imgl.shape) == 2:
                imgl = np.stack([imgl] * 3, 2)
            imgr = imgreader.imread(self.imgr_list[index])
            if len(imgr.shape) == 2:
                imgr = np.stack([imgr] * 3, 2)

            imglist = [imgl, imgl[:, ::-1, :], imgr, imgr[:, ::-1, :]]
            for i in range(len(imglist)):
                imglist[i] = (imglist[i] - 127.5) / 128.0
                imglist[i] = imglist[i].transpose(2, 0, 1)

            imgs = [img.astype('float32') for img in imglist]
            yield imgs

    def __len__(self):
        return len(self.imgl_list)


if __name__ == '__main__':
    pass
