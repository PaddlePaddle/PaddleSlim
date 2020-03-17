"""Define some functions to compute the importance of structure to be pruned.
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
from ..common import get_logger

__all__ = ["l1_norm"]

_logger = get_logger(__name__, level=logging.INFO)


def l1_norm(group):
    """Compute l1-norm scores of parameter on given axis.

    This function return a list of parameters' l1-norm scores on given axis.
    Each element of list is a tuple with format (name, axis, score) in which 'name' is parameter's name
    and 'axis' is the axis reducing on and `score` is a np.array storing the l1-norm of strucure on `axis`.

    Args:
       group(list): A group of parameters. The first parameter of the group is convolution layer's weight
                    while the others are parameters affected by pruning the first one. Each parameter in group
                    is represented as tuple '(name, values, axis)' in which `name` is the parameter's name and
                    and `values` is the values of parameter and `axis` is the axis reducing on pruning on.
    Returns:
       list: A list of tuple storing l1-norm on given axis.
    """
    scores = []
    for name, value, axis in group:

        reduce_dims = [i for i in range(len(value.shape)) if i != axis]
        score = np.sum(np.abs(value), axis=tuple(reduce_dims))
        scores.append((name, axis, score))

    return scores
