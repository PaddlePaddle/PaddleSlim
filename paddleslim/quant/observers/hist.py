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

from .base_hist import BaseHistObserver
from paddle.quantization.factory import ObserverFactory


class HistObserver(ObserverFactory):
    r"""
    It collects tensor values into a histogram. And calculate quantization parameters
    based on a percent ratio.

    Args:
        quant_bits (int): The number of bits for quantization.
        bins_count(int): The number of equal-width bins.
        percent(float): The percentage of bins that are retained when clipping the outliers.
        sign (bool): Whether the quantized integer includes a sign.
        symmetric (bool): Whether it is symmetric quantization. the quantization is symmetric.
        In symmetric quantization, the range of floating point values is relaxed to be symmetric
        around zero and the zero-point is always 0.
        

    Examples:
       .. code-block:: python

            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import HistObserver
            quanter = HistObserver()
            q_config = QuantConfig(activation=quanter, weight=quanter)
    """

    def __init__(self,
                 quant_bits=8,
                 bins_count=2048,
                 percent=0.999,
                 sign=True,
                 symmetric=True):
        super(HistObserver, self).__init__(
            quant_bits=quant_bits,
            bins_count=bins_count,
            percent=percent,
            sign=sign,
            symmetric=symmetric)

    def _get_class(self):
        return PercentHistObserverLayer


class PercentHistObserverLayer(BaseHistObserver):
    r"""
    It collects tensor values into a histogram. And calculate quantization parameters
    based on a percent ratio.
    """

    def __init__(self,
                 layer,
                 quant_bits=8,
                 bins_count=2048,
                 percent=0.999,
                 sign=True,
                 symmetric=True):
        super(PercentHistObserverLayer, self).__init__(
            quant_bits=quant_bits,
            bins_count=bins_count,
            sign=sign,
            symmetric=symmetric)

        self._percent = percent

    def _cal_min_max_by_percent(self):
        hist = self._hist / np.sum(self._hist, dtype=np.float64)
        cumsumed_hist = np.cumsum(hist)
        max_idx = np.argwhere(cumsumed_hist >= self._percent)[0]
        min_idx = np.argwhere(cumsumed_hist >= (1 - self._percent))[0]
        bin_width = (self._hist_max - self._hist_min) / hist.shape[0]
        _max = self._hist_min + float((max_idx - 0.5) * bin_width)
        _min = self._hist_min + float((min_idx - 0.5) * bin_width)
        return _min, _max

    def cal_min_max(self):
        return self._cal_min_max_by_percent()
