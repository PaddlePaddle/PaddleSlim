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
def default_idx_selector(group, scores, ratios):
    """Get the pruned indexes by scores of master tensor.

    This function return a list of parameters' pruned indexes on given axis.
    Each element of list is a tuple with format (name, axis, indexes)
    in which 'name' is parameter's name and 'axis' is the axis pruning on and
    `indexes` is indexes to be pruned.

    Args:
       group(Group): A group of pruning operations.
       scores(dict): The key is name of tensor, the value is a dict with axis as key and scores as value.
       ratios(doct): The pruned ratio of each tensor. The key is name of tensor and the value is the pruned ratio. 
     
    Returns:

       list: pruned indexes with format (name, axis, pruned_indexes).

    """
    # sort channels by the master convolution's score
    name = group.master["name"]
    axis = group.master["axis"]
    score = scores[name][axis]

    # get max convolution groups attribution
    max_groups = 1
    for prune_info in group.all_prune_info():
        groups = prune_info.op.attr("groups")
        if groups is not None and groups > max_groups:
            max_groups = groups
    if max_groups > 1:
        score = score.reshape([groups, -1])
        group_size = score.shape[1]
        # get score for each group of channels
        score = np.mean(score, axis=1)
    sorted_idx = score.argsort()
    ratio = ratios[name]
    pruned_num = int(round(len(sorted_idx) * ratio))
    pruned_idx = sorted_idx[:pruned_num]
    # convert indexes of channel groups to indexes of channels.
    if max_groups > 1:
        correct_idx = []
        for idx in pruned_idx:
            for offset in range(groups_size):
                correct_idx.append(idx * groups_size + offset)
        pruned_idx = correct_idx[:]
    ret = []
    for _prune_info in group.all_prune_info():
        ret.append((_prune_info.name, _prune_info.axis[0], pruned_idx))
    return ret


@IDX_SELECTOR.register
def optimal_threshold(group, scores, ratios):
    """Get the pruned indexes by scores of master tensor.

    This function return a list of parameters' pruned indexes on given axis.
    Each element of list is a tuple with format (name, axis, indexes)
    in which 'name' is parameter's name and 'axis' is the axis pruning on and
    `indexes` is indexes to be pruned.

    Args:
       group(Group): A group of pruning operations.
       scores(dict): The key is name of tensor, the value is a dict with axis as key and scores as value.
       ratios(doct): The pruned ratio of each tensor. The key is name of tensor and the value is the pruned ratio. 
     
    Returns:
       list: pruned indexes with format (name, axis, pruned_indexes).
    """
    # sort channels by the master tensor
    name = group.master["name"]
    axis = group.master["axis"]
    score = scores[name][axis]
    ratio = ratios[name]

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
    for _prune_info in group.all_prune_info():
        idxs.append(
            (_prune_info.name, _prune_info.axis, _prune_info.pruned_idx))
    return idxs
