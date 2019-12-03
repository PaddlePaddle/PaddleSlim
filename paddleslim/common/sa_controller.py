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
import logging
import numpy as np
from .controller import EvolutionaryController
from log_helper import get_logger

__all__ = ["SAController"]

_logger = get_logger(__name__, level=logging.INFO)


class SAController(EvolutionaryController):
    """Simulated annealing controller."""

    def __init__(self,
                 range_table=None,
                 reduce_rate=0.85,
                 init_temperature=1024,
                 max_try_times=None,
                 init_tokens=None,
                 constrain_func=None):
        """Initialize.
        Args:
            range_table(list<int>): Range table.
            reduce_rate(float): The decay rate of temperature.
            init_temperature(float): Init temperature.
            max_try_times(int): max try times before get legal tokens.
            init_tokens(list<int>): The initial tokens.
            constrain_func(function): The callback function used to check whether the tokens meet constraint. None means there is no constraint. Default: None.
        """
        super(SAController, self).__init__()
        self._range_table = range_table
        assert isinstance(self._range_table, tuple) and (
            len(self._range_table) == 2)
        self._reduce_rate = reduce_rate
        self._init_temperature = init_temperature
        self._max_try_times = max_try_times
        self._reward = -1
        self._tokens = init_tokens
        self._constrain_func = constrain_func
        self._max_reward = -1
        self._best_tokens = None
        self._iter = 0

    def __getstate__(self):
        d = {}
        for key in self.__dict__:
            if key != "_constrain_func":
                d[key] = self.__dict__[key]
        return d

    def update(self, tokens, reward, iter):
        """
        Update the controller according to latest tokens and reward.
        Args:
            tokens(list<int>): The tokens generated in last step.
            reward(float): The reward of tokens.
        """
        iter = int(iter)
        if iter > self._iter:
            self._iter = iter
        temperature = self._init_temperature * self._reduce_rate**self._iter
        if (reward > self._reward) or (np.random.random() <= math.exp(
            (reward - self._reward) / temperature)):
            self._reward = reward
            self._tokens = tokens
        if reward > self._max_reward:
            self._max_reward = reward
            self._best_tokens = tokens
        _logger.info(
            "Controller - iter: {}; current_reward: {}; current tokens: {}".
            format(self._iter, self._reward, self._tokens))

    def next_tokens(self, control_token=None):
        """
        Get next tokens.
        """
        if control_token:
            tokens = control_token[:]
        else:
            tokens = self._tokens
        new_tokens = tokens[:]
        index = int(len(self._range_table[0]) * np.random.random())
        new_tokens[index] = np.random.randint(self._range_table[0][index],
                                              self._range_table[1][index])
        _logger.debug("change index[{}] from {} to {}".format(index, tokens[
            index], new_tokens[index]))
        if self._constrain_func is None or self._max_try_times is None:
            return new_tokens
        for _ in range(self._max_try_times):
            if not self._constrain_func(new_tokens):
                index = int(len(self._range_table[0]) * np.random.random())
                new_tokens = tokens[:]
                new_tokens[index] = np.random.randint(
                    self._range_table[0][index],
                    self._range_table[1][index])
            else:
                break
        return new_tokens
