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
import numpy as np

from paddle.quantization.base_observer import BaseObserver
from paddle.quantization.factory import ObserverFactory


class HistObserver(ObserverFactory):
    r"""
    It collects maximum absolute values of target tensor.

    Args:
        bit_length(int, optional): Number of bits to represent an quantized integer in binary.
        dtype(str, optional): The data type of input tensor.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
       .. code-block:: python

            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
            quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.99)
            q_config = QuantConfig(activation=quanter, weight=quanter)
    """

    def __init__(self, quant_bits=8, percent=0.999):
        super(HistObserver, self).__init__(
            quant_bits=quant_bits, percent=percent)

    def _get_class(self):
        return HistObserverLayer


class HistObserverLayer(BaseObserver):
    """
    Per-tensor abs max quantizer.
    """

    def __init__(self,
                 layer,
                 quant_bits=8,
                 percent=0.999,
                 bin_count=2048,
                 init_steps=100):
        super(HistObserverLayer, self).__init__()
        self._quant_bits = quant_bits
        self._percent = percent
        self._bin_count = bin_count
        self._upsample_bin_count = 0

        self._min_max_values = None
        self._hists = None

    def _min_max(self, tensor):
        """" """
        return float(paddle.min(tensor).numpy()), float(
            paddle.max(tensor).numpy())

    def _init_hists(self, inputs):
        hists = []
        for _inp in inputs:
            _min, _max = self._min_max(_inp)
            hist = None
            if _max > _min:
                hist, _ = np.histogram(
                    _inp.numpy(), range=(_min, _max), bins=self._bin_count)
                hist.astype(np.float32)
            hists.append(hist)
        return hists

    def forward(self, inputs):
        """ Calculate forward pass.
        """
        inputs = self._to_list(inputs)
        if self._min_max_values is None:
            self._min_max_values = [self._min_max(_inp) for _inp in inputs]
            self._hists = self._init_hists(inputs)
        else:
            for idx, _inp in enumerate(inputs):

                new_abs_max, new_hist = self.update_min_max_and_hist(
                    _inp,
                    self._min_max_values[idx],
                    self._hists[idx],
                    self._bin_count,
                    self._upsample_bin_count, )
                self._min_max_values[idx] = new_abs_max
                self._hists[idx] = new_hist

        return inputs

    def merge_max_value(old, new):
        """
        Merge the max element one by one in two lists.
        """
        assert isinstance(old, list) and isinstance(new, list)
        if old != []:
            assert len(old) == len(new)
            for i in range(len(old)):
                assert type(old[i]) == type(new[i])
                if isinstance(old[i], list):
                    new[i] = merge_max_value(old[i], new[i])
                else:
                    new[i] = old[i] if new[i] < old[i] else new[i]
        return new

    def _to_list(self, inputs):
        if isinstance(inputs, paddle.Tensor):
            return [inputs]
        assert isinstance(inputs, (list, tuple))
        return inputs

    def update_min_max_and_hist(self, tensor, origin_min_max, origin_hist,
                                bins_count, upsample_bins_count):

        _origin_min, _origin_max = origin_min_max
        _new_min, _new_max = self._min_max(tensor)

        if (_new_max - _new_min) == 0.0:
            return origin_min_max, origin_hist
        elif _origin_max - _origin_min == 0.0:
            new_hist, _ = np.histogram(
                paddle.abs(tensor).numpy(),
                range=(_new_min, _new_max),
                bins=bins_count)
            new_hist = new_hist.astype(np.float32)
            return (_new_min, _new_max), new_hist
        elif _new_max <= _origin_max and _new_min >= _origin_min:
            new_hist, _ = np.histogram(
                paddle.abs(tensor).numpy(),
                range=(_origin_min, _origin_max),
                bins=bins_count)
            new_hist = new_hist.astype(np.float32)
            new_hist += origin_hist
            return origin_min_max, new_hist
        else:
            _new_min = min(_new_min, _origin_min)
            _new_max = min(_new_max, _origin_max)
            _new_min, _new_max, downsample_bins_count, start_bin_idx = self._relax_min_max(
                _new_min, _new_max, _origin_min, _origin_max, bins_count,
                upsample_bins_count)

            new_hist, _ = np.histogram(
                paddle.abs(tensor).numpy(),
                range=(_new_min, _new_max),
                bins=bins_count)

            merged_histogram = self._merge_histograms(
                new_hist, origin_hist, upsample_bins_count,
                downsample_bins_count, start_bin_idx, bins_count)
            return (_new_min, _new_max), new_hist

    def _merge_histograms(
            self,
            new_hist: np.Array,
            origin_hist: np.Array,
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

    def cal_thresholds(self):
        """ Compute thresholds for MAX function.
        """
        self.thresholds = self.abs_max_val

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
        return self.abs_max_val

    def zero_points(self):
        """ Return output zero points.
        """
        return None
