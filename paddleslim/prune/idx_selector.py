"""Define some functions to sort substructures of parameter by importance.
"""
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

import logging
import numpy as np
from ..core import GraphWrapper
from ..common import get_logger
from ..core import Registry

__all__ = ["IDX_SELECTOR"]

IDX_SELECTOR = Registry('idx_selector')


@IDX_SELECTOR.register
def default_idx_selector(group, ratio):
    """Get the pruned indexes by given ratio.

    This function return a list of parameters' pruned indexes on given axis.
    Each element of list is a tuple with format (name, axis, indexes) in which 'name' is parameter's name
    and 'axis' is the axis pruning on and `indexes` is indexes to be pruned.

    Args:
       group(list): A group of parameters. The first parameter of the group is convolution layer's weight
                    while the others are parameters affected by pruning the first one. Each parameter in group
                    is represented as tuple '(name, axis, score)' in which `name` is the parameter's name and
                    `axis` is the axis pruning on and `score` is a np.array storing the importance of strucure
                    on `axis`. Show as below:

                    .. code-block: text

                       [("conv1_weights", 0, [0.7, 0.5, 0.6]), ("conv1_bn.scale", 0, [0.1, 0.2, 0.4])]

                    The shape of "conv1_weights" is `[out_channel, in_channel, filter_size, filter_size]`, so
                    `[0.7, 0.5, 0.6]` are the importance sores of each output channel in "conv1_weights"
                    while axis is 0.
     
    Returns:

       list: pruned indexes

    """
    name, axis, score, _ = group[
        0]  # sort channels by the first convolution's score
    sorted_idx = score.argsort()

    pruned_num = int(round(len(sorted_idx) * ratio))
    pruned_idx = sorted_idx[:pruned_num]
    idxs = []
    for name, axis, score, transforms in group:
        idxs.append((name, axis, pruned_idx, transforms))
    return idxs


@IDX_SELECTOR.register
def optimal_threshold(group, ratio):
    """Get the pruned indexes by given ratio.

    This function return a list of parameters' pruned indexes on given axis.
    Each element of list is a tuple with format (name, axis, indexes) in which 'name' is parameter's name
    and 'axis' is the axis pruning on and `indexes` is indexes to be pruned.

    Args:
       group(list): A group of parameters. The first parameter of the group is convolution layer's weight
                    while the others are parameters affected by pruning the first one. Each parameter in group
                    is represented as tuple '(name, axis, score)' in which `name` is the parameter's name and
                    `axis` is the axis pruning on and `score` is a np.array storing the importance of strucure
                    on `axis`. Show as below:

                    .. code-block: text

                       [("conv1_weights", 0, [0.7, 0.5, 0.6]), ("conv1_bn.scale", 0, [0.1, 0.2, 0.4])]

                    The shape of "conv1_weights" is `[out_channel, in_channel, filter_size, filter_size]`, so
                    `[0.7, 0.5, 0.6]` are the importance sores of each output channel in "conv1_weights"
                    while axis is 0.
     
    Returns:

       list: pruned indexes

    """
    name, axis, score, _ = group[
        0]  # sort channels by the first convolution's score

    score[score < 1e-18] = 1e-18
    score_sorted = np.sort(score)
    score_square = score_sorted**2
    total_sum = score_square.sum()
    acc_sum = 0
    for i in range(score_square.size):
        acc_sum += score_square[i]
        if acc_sum / total_sum > ratio:
            break
    th = (score_sorted[i - 1] + score_sorted[i]) / 2 if i > 0 else 0

    pruned_idx = np.squeeze(np.argwhere(score < th))

    idxs = []
    for name, axis, score, transforms in group:
        idxs.append((name, axis, pruned_idx, transforms))
    return idxs
