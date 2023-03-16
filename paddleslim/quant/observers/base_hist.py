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
from typing import Tuple
import paddle
import numpy as np

from .uniform import UniformObserver


class BaseHistObserver(UniformObserver):
    """
    It is a base class of histogram observers defined some functions to
    collects the values of multi batches to a histogram.
    Args:
        quant_bits (int): The number of bits for quantization.
        sign (bool): Whether the quantized integer includes a sign.
        symmetric (bool): Whether it is symmetric quantization. the quantization is symmetric.
        In symmetric quantization, the range of floating point values is relaxed to be symmetric
        around zero and the zero-point is always 0.
        bins_count(int): The number of equal-width bins.
    """

    def __init__(self, quant_bits=8, bins_count=2048, sign=True,
                 symmetric=True):
        super(BaseHistObserver, self).__init__(
            quant_bits=quant_bits,
            sign=sign,
            symmetric=symmetric, )
        self._bin_count = bins_count
        self._upsample_bin_count = 64

        self._hist_min = None
        self._hist_max = None
        self._hist = None

    def _min_max(self, tensor):
        """" Get the min and max value of a tensor.
        """
        return float(paddle.min(tensor).numpy()), float(
            paddle.max(tensor).numpy())

    def _init_hists(self, inputs):
        """" Initialize the histogram instance based on a tensor.
        """
        _min, _max = self._min_max(inputs)
        hist = None
        if _max > _min:
            hist, _ = np.histogram(
                inputs.numpy(), range=(_min, _max), bins=self._bin_count)
            hist.astype(np.float32)
        return hist

    def forward(self, inputs):
        self._scale = None
        self._zero_point = None
        self._min = None
        self._max = None

        if self._hist_min is None or self._hist_max is None:
            self._hist_min, self._hist_max = self._min_max(inputs)
            self._hist = self._init_hists(inputs)
        else:
            new_min, new_max, new_hist = self._update_min_max_and_hist(
                inputs,
                self._hist_min,
                self._hist_max,
                self._hist,
                self._bin_count,
                self._upsample_bin_count, )
            self._hist_min, self._hist_max = new_min, new_max
            self._hist = new_hist
        return inputs

    def _update_min_max_and_hist(self, tensor, origin_min, origin_max,
                                 origin_hist, bins_count, upsample_bins_count):
        """ Update the histogram and its range based on the values of the target tensor.
        Args:
            tensor: The tensor used to update the histogram.
            origin_min(float): The minimum of the original histogram's range.
            origin_max(float): The max of the original histogram's range.
            origin_hist: The original histogram.
            bins_count(int): The number of histogram bins.
            upsample_bins_count(int): The number of upsampled bins used to extend the histogram.
        """

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
        new_hist = new_hist.astype(np.float32)
        new_hist += sampled_hist.astype(np.float32)
        return new_hist

    def _relax_min_max(self, new_min, new_max, origin_min, origin_max,
                       bins_count,
                       upsample_bins_count) -> Tuple[float, float, int, int]:
        _bin_width = (origin_max - origin_min) / (
            bins_count * upsample_bins_count)
        downsample_bins_count = int(
            np.ceil((new_max - new_min) / (bins_count * _bin_width)))
        error = downsample_bins_count * bins_count * _bin_width - (
            new_max - new_min)
        new_max += error
        start_bin_idx = round((origin_min - new_min) / _bin_width)
        return new_min, new_max, downsample_bins_count, start_bin_idx

    @abc.abstractmethod
    def cal_min_max(self) -> Tuple[float, float]:
        """ Calculate the minimum and maximum based on the histogram. """
        raise NotImplementedError("Please implement the abstract method.")

    def cal_thresholds(self):
        assert self._hist is not None
        self._min, self._max = self.cal_min_max()
        self._scale, self._zero_point = self.cal_scales_zero_points()

    def min_value(self) -> float:
        return self._min

    def max_value(self) -> float:
        return self._max

    def bit_length(self):
        return self._quant_bits

    def quant_axis(self):
        return -1

    def scales(self):
        if self._scale is None:
            self.cal_thresholds()
        return self._scale

    def zero_points(self):
        if self._zero_point is None:
            self.cal_thresholds()
        return self._zero_point
