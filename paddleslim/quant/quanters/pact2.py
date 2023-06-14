# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import random
import numpy as np
from paddle.utils import unique_name
from paddle.nn.initializer import Constant, Assign
from paddle.quantization.factory import QuanterFactory
from paddle.quantization.base_quanter import BaseQuanter
from .base_fake_quanter import BaseFakeQuanterLayer

PACT2_RANGE_INIT = 8.0


class PACT2Quanter(QuanterFactory):
    def __init__(self,
                 extrema_mode='min_max',
                 update_range=True,
                 clip_range=None,
                 quant_bits=8,
                 sign=True):
        super(PACT2Quanter, self).__init__(
            extrema_mode=extrema_mode,
            update_range=update_range,
            clip_range=clip_range,
            quant_bits=quant_bits,
            sign=sign)

    def _get_class(self):
        return PACT2QuanterLayer


class PACT2QuanterLayer(BaseFakeQuanterLayer):
    def __init__(self,
                 layer,
                 extrema_mode='min_max',
                 update_range=True,
                 clip_range=None,
                 quant_bits=8,
                 sign=True):
        super(PACT2QuanterLayer, self).__init__(quant_bits, sign)
        self._extrema_mode = extrema_mode
        self._update_range = update_range
        self._qmin, self._qmax = self.qmin_qmax

        clip_init = max(abs(np.array(clip_range))) if (
            clip_range is not None) else PACT2_RANGE_INIT
        clip_init2 = np.power(2.0, np.ceil(np.log2(clip_init)))

        value_prefix = ("{}.pact2_range".format(layer.full_name()))
        name = unique_name.generate(value_prefix)

        clip_attr = paddle.ParamAttr(
            name=name,
            initializer=Assign([-clip_init2, clip_init2]),
            trainable=False)

        self.clip_value = self.create_parameter(
            shape=[2], attr=clip_attr, dtype='float32')

        scale_prefix = ("{}.scale".format(layer.full_name()))
        scale_name = unique_name.generate(scale_prefix)

        scale_attr = paddle.ParamAttr(
            name=scale_name, initializer=Constant(1.0), trainable=True)
        self._scale = self.create_parameter(
            shape=[1], attr=scale_attr, dtype='float32')

        self._eps = 2.0**(-16.0)
        self._range_update_factor_min = 0.001
        self._current_iters = 0

    def forward(self, activation):
        self._current_iters += 1
        if self._update_range:
            with paddle.no_grad():
                self._update_clip_range(activation)
        clips = self._get_clips_value()
        act = paddle.clip(activation, clips[0], clips[1])
        self._scale.set_value(clips[1])
        return self._quant_dequant(act)

    def _update_clip_range(self, act):
        clip_min, clip_max = self.clip_value[0], self.clip_value[1]

        min_value, max_value = self._cal_extrema(act)

        # exponential moving average update
        update_factor = 1.0 / self._current_iters
        update_factor = max(update_factor, self._range_update_factor_min)
        clip_max = clip_max * (1 - update_factor) + max_value * update_factor
        clip_min = clip_min * (1 - update_factor) + min_value * update_factor

        self.clip_value.set_value(paddle.concat([clip_min, clip_max]))
        self.clip_value[0].set_value(clip_min)
        self.clip_value[1].set_value(clip_max)

    def _cal_extrema(self, src):
        if self._extrema_mode == 'min_max':
            min_value = src.min()
            max_value = src.max()
        elif self._extrema_mode == 'histogram':
            hist_array, min_value, max_value, mult_factor, offset = self._tensor_histogram(
                src)
            if hist_array is None:
                return min_value, max_value

            new_mn_scaled, new_mx_scaled = self._extrema_hist_search(hist_array)
            new_mn = (new_mn_scaled / mult_factor) + offset
            new_mx = (new_mx_scaled / mult_factor) + offset

            new_mn = paddle.max(min_value, new_mn)
            new_mx = paddle.min(max_value, new_mx)
            return new_mn, new_mx

        elif self._extrema_mode == 'sigma':
            mean = src.mean()
            std = src.std()
            min_value = mean - 0.5 * std
            max_value = mean + 0.5 * std

        return min_value, max_value

    def _get_clips_value(self):
        if self._sign:
            clip_max = paddle.max(paddle.abs(self.clip_value))
        else:
            clip_max = paddle.abs(self.clip_value[1])
        clip_max = paddle.clip(clip_max, min=self._eps)
        # pow2
        clip_max2 = clip_max + (paddle.pow(
            paddle.to_tensor(2.), paddle.ceil(paddle.log2(clip_max))) -
                                clip_max).detach()
        clip_min2 = (-clip_max2 if self._sign else clip_max2 * 0.0)
        return (clip_min2, clip_max2)

    def _quant_dequant(self, x):
        s = self._scale / self._qmax
        # pow2
        s = paddle.pow(paddle.to_tensor(2.), paddle.floor(paddle.log2(s)))
        quant_x = paddle.clip(paddle.floor(x / s + 0.5), self._qmin, self._qmax)
        dequant_x = s * quant_x
        return x + (dequant_x - x).detach()

    def bit_length(self):
        """ Return the bit length of quantized data.
        """
        return self._quant_bits

    def quant_axis(self):
        """ Return quantization axis.
        """
        return -1

    def scales(self):
        """ Return output scales.
        """
        return self._scale

    def zero_points(self):
        """ Return output zero points.
        """
        if self._zero_point is None:
            if self._symmetric:
                if self._sign:
                    self._zero_point = 0
                else:
                    self._zero_point = (self._qmax + self._qmin) / 2
        return self._zero_point

    def _tensor_histogram(self, src, fast_mode=True):
        # downsample for fast_mode
        fast_stride = 2
        fast_stride2 = fast_stride * 2
        if fast_mode and len(src.shape) == 4 and (
                src.shape[2] > fast_stride2) and (src.shape[3] > fast_stride2):
            r_start = random.randint(0, fast_stride - 1)
            c_start = random.randint(0, fast_stride - 1)
            src = src[..., r_start::fast_stride, c_start::fast_stride]

        mn = src.min()
        mx = src.max()
        if mn == 0 and mx == 0:
            return None, mn, mx, 1.0, 0.0

        num_bins = 255.0
        cum_freq = float(100.0)
        offset = mn
        range_val = paddle.abs(mx - mn)
        mult_factor = (num_bins / range_val)
        tensor_int = (src.flatten() - offset) * mult_factor
        tensor_int = paddle.round(tensor_int)

        hist = np.bincount(tensor_int.numpy().astype(np.int32))
        hist_sum = np.sum(hist)
        hist_array = hist.astype(np.float32) * cum_freq / float(hist_sum)
        return hist_array, mn, mx, mult_factor, offset

    def _extrema_hist_search(self, hist_array, range_shrink_percentile=0.01):
        new_mn_scaled = 0
        new_mx_scaled = len(hist_array) - 1
        hist_sum_left = 0.0
        hist_sum_right = 0.0
        for h_idx in range(len(hist_array)):
            r_idx = len(hist_array) - 1 - h_idx
            hist_sum_left += hist_array[h_idx]
            hist_sum_right += hist_array[r_idx]
            if hist_sum_left < range_shrink_percentile:
                new_mn_scaled = h_idx
            if hist_sum_right < range_shrink_percentile:
                new_mx_scaled = r_idx
        return paddle.to_tensor(new_mn_scaled), paddle.to_tensor(new_mx_scaled)
