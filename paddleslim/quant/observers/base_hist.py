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

import abc
import paddle
import numpy as np

from .uniform import UniformObserver


class BaseHistObserver(UniformObserver):
    """
    Per-tensor abs max quantizer.
    """

    def __init__(self,
                 quant_bits=8,
                 bins_count=2048,
                 upsample_bins_count=64,
                 sign=True,
                 symmetric=True):
        super(BaseHistObserver, self).__init__(
            quant_bits=quant_bits,
            sign=sign,
            symmetric=symmetric, )
        self._bin_count = bins_count
        self._upsample_bin_count = upsample_bins_count

        self._hist_min = None
        self._hist_max = None
        self._hist = None

    def _min_max(self, tensor):
        """" """
        return float(paddle.min(tensor).numpy()), float(
            paddle.max(tensor).numpy())

    def _init_hists(self, inputs):
        _min, _max = self._min_max(inputs)
        hist = None
        if _max > _min:
            hist, _ = np.histogram(
                inputs.numpy(), range=(_min, _max), bins=self._bin_count)
            hist.astype(np.float32)
        return hist

    def forward(self, inputs):
        """ Calculate forward pass.
        """
        self._scale = None
        self._zero_point = None
        self._min = None
        self._max = None

        if self._hist_min is None or self._hist_max is None:
            self._hist_min, self._hist_max = self._min_max(inputs)
            self._hist = self._init_hists(inputs)
        else:
            new_min, new_max, new_hist = self.update_min_max_and_hist(
                inputs,
                self._hist_min,
                self._hist_max,
                self._hist,
                self._bin_count,
                self._upsample_bin_count, )
            self._hist_min, self._hist_max = new_min, new_max
            self._hist = new_hist
        print(f"self._hist: {self._hist}")
        return inputs

    def update_min_max_and_hist(self, tensor, origin_min, origin_max,
                                origin_hist, bins_count, upsample_bins_count):

        _origin_min, _origin_max = origin_min, origin_max
        _new_min, _new_max = self._min_max(tensor)

        if (_new_max - _new_min) == 0.0:
            return _origin_min, _origin_max, origin_hist
        elif _origin_max - _origin_min == 0.0:
            new_hist, _ = np.histogram(
                tensor.numpy(), range=(_new_min, _new_max), bins=bins_count)
            new_hist = new_hist.astype(np.float32)
            return _new_min, _new_max, new_hist
        elif _new_max <= _origin_max and _new_min >= _origin_min:
            new_hist, _ = np.histogram(
                tensor.numpy(),
                range=(_origin_min, _origin_max),
                bins=bins_count)
            new_hist = new_hist.astype(np.float32)
            new_hist += origin_hist
            return _origin_min, _origin_max, new_hist
        else:
            _new_min = min(_new_min, _origin_min)
            _new_max = max(_new_max, _origin_max)
            _new_min, _new_max, downsample_bins_count, start_bin_idx = self._relax_min_max(
                _new_min, _new_max, _origin_min, _origin_max, bins_count,
                upsample_bins_count)

            new_hist, _ = np.histogram(
                tensor.numpy(), range=(_new_min, _new_max), bins=bins_count)

            merged_histogram = self._merge_histograms(
                new_hist, origin_hist, upsample_bins_count,
                downsample_bins_count, start_bin_idx, bins_count)
            return _new_min, _new_max, merged_histogram

    def _merge_histograms(
            self,
            new_hist: np.ndarray,
            origin_hist: np.ndarray,
            upsample_bins_count: int,
            downsample_bins_count: int,
            start_bin_idx: int,
            bins_count: int, ):
        upsampled_histogram = np.repeat(origin_hist, upsample_bins_count)
        expanded_hist = np.zeros(
            (bins_count * downsample_bins_count), dtype=np.float32)
        expanded_hist[start_bin_idx:bins_count * upsample_bins_count +
                      start_bin_idx] = upsampled_histogram

        cumsumed_hist = np.cumsum(
            expanded_hist,
            dtype=np.float64)[downsample_bins_count - 1::downsample_bins_count]
        shift_cumsumed_hist = np.zeros((bins_count), dtype=np.float64)
        shift_cumsumed_hist[1:] = cumsumed_hist[0:-1]
        sampled_hist = (
            cumsumed_hist - shift_cumsumed_hist) / upsample_bins_count
        new_hist += sampled_hist.astype(np.float32)
        return new_hist

    def _relax_min_max(self, new_min, new_max, origin_min, origin_max,
                       bins_count, upsample_bins_count):
        _bin_width = (origin_max - origin_min) / (
            bins_count * upsample_bins_count)
        downsample_bins_count = np.ceil(
            (new_max - new_min) / (bins_count * _bin_width))
        error = downsample_bins_count * bins_count * _bin_width - (
            new_max - new_min)
        new_max += error
        start_bin_idx = round((origin_min - new_min) / _bin_width)
        return new_min, new_max, downsample_bins_count, start_bin_idx

    @abc.abstractmethod
    def cal_min_max(self):
        pass

    def cal_thresholds(self):
        """ Compute thresholds for MAX function.
        """
        assert self._hist is not None
        self._min, self._max = self.cal_min_max()
        self._scale, self._zero_point = self.cal_scales_zero_points()

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
        if self._scale is None:
            self.cal_thresholds()
        return self._scale

    def zero_points(self):
        """ Return output zero points.
        """
        if self._zero_point is None:
            self.cal_thresholds()
        return self._zero_point
