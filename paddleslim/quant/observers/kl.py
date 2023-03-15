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

import numpy as np
import math

from .base_hist import BaseHistObserver
from paddle.quantization.factory import ObserverFactory


class KLObserver(ObserverFactory):
    r""" 
    Calculate quantization parameters that minimize the Kullback–Leibler divergence
    between the distribution of floating values and the distribution of quantized
    floating values.

    Args:
        quant_bits (int): The number of bits for quantization.
        bins_count(int): The number of equal-width bins.

    Examples:
       .. code-block:: python

            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import KLObserver
            quanter = KLObserver()
            q_config = QuantConfig(activation=quanter, weight=quanter)
    """

    def __init__(self, quant_bits=8, bins_count=2048):
        super(KLObserver, self).__init__(
            quant_bits=quant_bits, bins_count=bins_count)

    def _get_class(self):
        return KLObserverLayer


class KLObserverLayer(BaseHistObserver):
    """
    Per-tensor KL observer.
    """

    def __init__(self, layer, quant_bits=8, bins_count=2048):
        super(KLObserverLayer, self).__init__(
            quant_bits=quant_bits,
            bins_count=bins_count,
            sign=True,
            symmetric=True)

    def _search_min_max_by_kl(self):
        bin_width = (self._hist_max - self._hist_min) / self._bin_count
        _max = cal_kl_threshold(self._hist, bin_width, self.bit_length())
        return 0., _max

    def cal_min_max(self):
        return self._search_min_max_by_kl()


def expand_quantized_bins(quantized_bins, reference_bins):
    '''
    Expand hist bins.
    '''
    expanded_quantized_bins = [0] * len(reference_bins)
    num_merged_bins = int(len(reference_bins) / len(quantized_bins))
    j_start = 0
    j_end = num_merged_bins
    for idx in range(len(quantized_bins)):
        zero_count = reference_bins[j_start:j_end].count(0)
        num_merged_bins = j_end - j_start
        if zero_count == num_merged_bins:
            avg_bin_ele = 0
        else:
            avg_bin_ele = quantized_bins[idx] / (
                num_merged_bins - zero_count + 0.0)
        for idx1 in range(j_start, j_end):
            expanded_quantized_bins[idx1] = (0 if reference_bins[idx1] == 0 else
                                             avg_bin_ele)
        j_start += num_merged_bins
        j_end += num_merged_bins
        if (idx + 1) == len(quantized_bins) - 1:
            j_end = len(reference_bins)
    return expanded_quantized_bins


def safe_entropy(reference_distr_P, P_sum, candidate_distr_Q, Q_sum):
    '''
    Calculate the entropy.
    '''
    assert len(reference_distr_P) == len(candidate_distr_Q)
    tmp_sum1 = 0
    tmp_sum2 = 0
    for idx in range(len(reference_distr_P)):
        p_idx = reference_distr_P[idx]
        q_idx = candidate_distr_Q[idx]
        if p_idx == 0:
            tmp_sum1 += 0
            tmp_sum2 += 0
        else:
            if q_idx == 0:
                _logger.error("Fatal error!, idx = " + str(idx) +
                              " qindex = 0! p_idx = " + str(p_idx))
            tmp_sum1 += p_idx * (math.log(Q_sum * p_idx))
            tmp_sum2 += p_idx * (math.log(P_sum * q_idx))
    return (tmp_sum1 - tmp_sum2) / P_sum


def cal_kl_threshold(hist, bin_width, bits):
    '''
    Using the KL-divergenc method to get the more precise threshold.

    Args:
        hist(List): The hist of the tensor.
        bin_width(float): The bin width for the hist.
        bits(int): The quantization bits.
    '''
    assert hist.ndim == 1
    hist_bins = hist.shape[0]
    starting_iter = int((hist_bins - 1) * 0.5)
    quant_range = 2**(bits - 1) - 1

    P_sum = np.sum(np.array(hist).ravel())
    min_kl_divergence = 0
    min_kl_index = 0
    kl_inited = False

    for i in range(starting_iter, hist_bins):
        reference_distr_P = hist[0:i].tolist()
        outliers_count = sum(hist[i:])
        if reference_distr_P[i - 1] == 0:
            continue
        reference_distr_P[i - 1] += outliers_count
        reference_distr_bins = reference_distr_P[:]
        candidate_distr_Q = hist[0:i].tolist()
        num_merged_bins = int(i / quant_range)
        candidate_distr_Q_quantized = [0] * quant_range
        j_start = 0
        j_end = num_merged_bins
        for idx in range(quant_range):
            candidate_distr_Q_quantized[idx] = sum(
                candidate_distr_Q[j_start:j_end])
            j_start += num_merged_bins
            j_end += num_merged_bins
            if (idx + 1) == quant_range - 1:
                j_end = i
        candidate_distr_Q = expand_quantized_bins(candidate_distr_Q_quantized,
                                                  reference_distr_bins)
        Q_sum = sum(candidate_distr_Q)
        kl_divergence = safe_entropy(reference_distr_P, P_sum,
                                     candidate_distr_Q, Q_sum)
        if not kl_inited:
            min_kl_divergence = kl_divergence
            min_kl_index = i
            kl_inited = True
        elif kl_divergence < min_kl_divergence:
            min_kl_divergence = kl_divergence
            min_kl_index = i
        else:
            pass
    if min_kl_index == 0:
        while starting_iter > 0:
            if hist[starting_iter] == 0:
                starting_iter -= 1
                continue
            else:
                break
        min_kl_index = starting_iter
    return (min_kl_index + 0.5) * bin_width
