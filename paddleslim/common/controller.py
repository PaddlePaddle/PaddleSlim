#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
"""The controller used to search hyperparameters or neural architecture"""

import copy
import math
import numpy as np

__all__ = ['EvolutionaryController']


class EvolutionaryController(object):
    """Abstract controller for all evolutionary searching method.
    """

    def update(self, tokens, reward):
        """Update the status of controller according current tokens and reward.

        Args:
            tokens(list<int>): A solution of searching task.
            reward(list<int>): The reward of tokens.
        """
        raise NotImplementedError('Abstract method.')

    def reset(self, range_table, constrain_func=None):
        """Reset the controller.

        Args:
            range_table(list<int>): It is used to define the searching space of controller.
                                    The tokens[i] generated by controller should be in [0, range_table[i]).
            constrain_func(function): It is used to check whether tokens meet the constraint.
                                     None means there is no constraint. Default: None.
        """
        raise NotImplementedError('Abstract method.')

    def next_tokens(self):
        """Generate new tokens.

        Returns:
            list<list>: The next searched tokens.
        """
        raise NotImplementedError('Abstract method.')
