# Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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

"""Random Erasing for image tensor"""

import random
import math
import paddle


def _get_pixels(per_pixel, rand_color, patch_size, dtype="float32"):
    if per_pixel:
        return paddle.normal(shape=patch_size).astype(dtype)
    if rand_color:
        return paddle.normal(shape=(patch_size[0], 1, 1)).astype(dtype)
    return paddle.zeros((patch_size[0], 1, 1)).astype(dtype)


class RandomErasing():
    """
    Args:
        prob: probability of performing random erasing
        min_area: Minimum percentage of erased area wrt input image area
        max_area: Maximum percentage of erased area wrt input image area
        min_aspect: Minimum aspect ratio of earsed area
        max_aspect: Maximum aspect ratio of earsed area
        mode: pixel color mode, in ['const', 'rand', 'pixel']
            'const' - erase block is constant valued 0 for all channels
            'rand'  - erase block is valued random color (same per-channel)
            'pixel' - erase block is vauled random color per pixel
        min_count: Minimum # of ereasing blocks per image.
        max_count: Maximum # of ereasing blocks per image. Area per box is scaled by count
                   per-image count is randomly chosen between min_count to max_count
    """
    def __init__(self, prob=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
                 mode='const', min_count=1, max_count=None, num_splits=0):
        self.prob = prob
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == "rand":
            self.rand_color = True
        elif mode == "pixel":
            self.per_pixel = True
        else:
            assert not mode or mode == "const"

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.prob:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attemp in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top+h, left:left+w] = _get_pixels(
                                self.per_pixel, self.rand_color, (chan, h, w),
                                dtype=dtype)
                    break

    def __call__(self, inputs):
        if len(inputs.shape) == 3:
            self._erase(inputs, *inputs.shape, inputs.dtype)
        else:
            batch_size, chan, img_h, img_w = inputs.shape
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(inputs[i], chan, img_h, img_w, inputs.dtype)
        return inputs



#def main():
#    re = RandomErasing(prob=1.0, min_area=0.2, max_area=0.6, mode='rand')
#    #re = RandomErasing(prob=1.0, min_area=0.2, max_area=0.6, mode='const')
#    #re = RandomErasing(prob=1.0, min_area=0.2, max_area=0.6, mode='pixel')
#    import PIL.Image as Image
#    import numpy as np
#    paddle.set_device('cpu')
#    img = paddle.to_tensor(np.asarray(Image.open('./lenna.png'))).astype('float32')
#    img = img / 255.0
#    img = paddle.transpose(img, [2, 0, 1])
#    new_img = re(img)
#    new_img = new_img * 255.0
#    new_img = paddle.transpose(new_img, [1, 2, 0])
#    new_img = new_img.cpu().numpy()
#    new_img = Image.fromarray(new_img.astype('uint8'))
#    new_img.save('./res.png')
#
#
#
#if __name__ == "__main__":
#    main()
