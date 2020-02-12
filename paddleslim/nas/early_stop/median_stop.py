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

import logging
from .early_stop import EarlyStopBase
from ...common.log_helper import get_logger

__all__ = ['MedianStop']

_logger = get_logger(__name__, level=logging.INFO)


class MedianStop(EarlyStopBase):
    """
    Median Stop, reference:
    Args:
        strategy<str>: the stategy of search.
        start_epoch<int>: which step to start early stop algorithm.
    """

    def __init__(self, strategy, start_epoch, mode='maximize'):
        self.start_epoch = start_epoch
        self._running_history = dict()
        self._completed_avg_history = dict()
        self._strategy = strategy
        self._mode = mode

    def _update_data(self, exp_name, result):
        if exp_name not in self._running_history.keys():
            self._running_history[exp_name] = []
        self._running_history[exp_name].append(result)

    def _convert_running2completed(self, exp_name, status):
        """
        Convert experiment record from running to complete.

        Args:
           exp_name<str>: the name of experiment.
           status<str>: the status of this experiment.
        """
        if exp_name in self._running_history:
            if status == "GOOD":
                count = 0
                history_sum = 0
                for res in self._running_history[exp_name]:
                    count += 1
                    history_sum += res
                    self.completed_avg_history[exp_name].append(history_sum /
                                                                count)
            self._running_history.pop(exp_name)

    def get_status(self, step, result, epochs):
        """ 
        Get current experiment status
        
        Args:
            step: step in this client.
            result: the result of this epoch.
            epochs: whole epochs.

        Return:
            the status of this experiment.
        """
        exp_name = self._strategy._client_name + str(step)
        self._update_data(exp_name, result)

        _logger.info("running history after update data: {}".format(
            self._running_history))

        curr_step = len(self._running_history[exp_name])
        status = "GOOD"
        if curr_step < self.start_epoch:
            return status

        res_same_step = []
        if len(self._completed_avg_history) == 0:
            for exp in self._running_history.keys():
                if curr_step <= len(self._running_history[exp]):
                    res_same_step.append(self._running_history[exp][curr_step -
                                                                    1])

        for exp in self._completed_avg_history.keys():
            if curr_step <= len(self._completed_avg_history[exp]):
                res_same_step.append(self._completed_avg_history[exp][curr_step
                                                                      - 1])

        _logger.info(res_same_step)
        if res_same_step:
            res_same_step.sort()

            if self._mode == 'maximize' and result < res_same_step[(
                    len(res_same_step) - 1) // 2]:
                status = "BAD"

            if self._mode == 'minimize' and result > res_same_step[len(
                    res_same_step) // 2]:
                status = "BAD"

        if curr_step == epochs:
            self._convert_running2completed(exp_name, status)

        return status
