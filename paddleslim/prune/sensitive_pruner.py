# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import os
import logging
import copy
from scipy.optimize import leastsq
import numpy as np
import paddle.fluid as fluid
from ..common import get_logger
from .sensitive import sensitivity
from .sensitive import flops_sensitivity, get_ratios_by_loss
from ..analysis import flops
from .pruner import Pruner

__all__ = ["SensitivePruner"]

_logger = get_logger(__name__, level=logging.INFO)


class SensitivePruner(object):
    """
    Pruner used to prune parameters iteratively according to sensitivities
    of parameters in each step.

    Args:
        place(fluid.CUDAPlace | fluid.CPUPlace): The device place where
            program execute.
        eval_func(function): A callback function used to evaluate pruned
            program. The argument of this function is pruned program.
            And it return a score of given program.
        scope(fluid.scope): The scope used to execute program.
    """

    def __init__(self, place, eval_func, scope=None, checkpoints=None):
        self._eval_func = eval_func
        self._iter = 0
        self._place = place
        self._scope = fluid.global_scope() if scope is None else scope
        self._pruner = Pruner()
        self._checkpoints = checkpoints

    def save_checkpoint(self, train_program, eval_program):
        checkpoint = os.path.join(self._checkpoints, str(self._iter - 1))
        exe = fluid.Executor(self._place)
        fluid.io.save_persistables(
            exe, checkpoint, main_program=train_program, filename="__params__")

        with open(checkpoint + "/main_program", "wb") as f:
            f.write(train_program.desc.serialize_to_string())
        with open(checkpoint + "/eval_program", "wb") as f:
            f.write(eval_program.desc.serialize_to_string())

    def restore(self, checkpoints=None):

        exe = fluid.Executor(self._place)
        checkpoints = self._checkpoints if checkpoints is None else checkpoints
        _logger.info("check points: {}".format(checkpoints))
        main_program = None
        eval_program = None
        if checkpoints is not None:
            cks = [dir for dir in os.listdir(checkpoints)]
            if len(cks) > 0:
                latest = max([int(ck) for ck in cks])
                latest_ck_path = os.path.join(checkpoints, str(latest))
                self._iter += 1

                with open(latest_ck_path + "/main_program", "rb") as f:
                    program_desc_str = f.read()
                main_program = fluid.Program.parse_from_string(
                    program_desc_str)

                with open(latest_ck_path + "/eval_program", "rb") as f:
                    program_desc_str = f.read()
                eval_program = fluid.Program.parse_from_string(
                    program_desc_str)

                with fluid.scope_guard(self._scope):
                    fluid.io.load_persistables(exe, latest_ck_path,
                                               main_program, "__params__")
                _logger.info("load checkpoint from: {}".format(latest_ck_path))
                _logger.info("flops of eval program: {}".format(
                    flops(eval_program)))
        return main_program, eval_program, self._iter

    def greedy_prune(self,
                     train_program,
                     eval_program,
                     params,
                     pruned_flops_rate,
                     topk=1):

        sensitivities_file = "greedy_sensitivities_iter{}.data".format(
            self._iter)
        with fluid.scope_guard(self._scope):
            sensitivities = flops_sensitivity(
                eval_program,
                self._place,
                params,
                self._eval_func,
                sensitivities_file=sensitivities_file,
                pruned_flops_rate=pruned_flops_rate)
        _logger.info(sensitivities)
        params, ratios = self._greedy_ratio_by_sensitive(sensitivities, topk)

        _logger.info("Pruning: {} by {}".format(params, ratios))
        pruned_program = self._pruner.prune(
            train_program,
            self._scope,
            params,
            ratios,
            place=self._place,
            only_graph=False)
        pruned_val_program = None
        if eval_program is not None:
            pruned_val_program = self._pruner.prune(
                eval_program,
                self._scope,
                params,
                ratios,
                place=self._place,
                only_graph=True)
        self._iter += 1
        return pruned_program, pruned_val_program

    def prune(self, train_program, eval_program, params, pruned_flops):
        """
        Pruning parameters of training and evaluation network by sensitivities in current step.

        Args:
            train_program(fluid.Program): The training program to be pruned.
            eval_program(fluid.Program): The evaluation program to be pruned. And it is also used to calculate sensitivities of parameters.
            params(list<str>): The parameters to be pruned.
            pruned_flops(float): The ratio of FLOPS to be pruned in current step.

        Returns:
            tuple: A tuple of pruned training program and pruned evaluation program.
        """
        _logger.info("Pruning: {}".format(params))
        sensitivities_file = "sensitivities_iter{}.data".format(self._iter)
        with fluid.scope_guard(self._scope):
            sensitivities = sensitivity(
                eval_program,
                self._place,
                params,
                self._eval_func,
                sensitivities_file=sensitivities_file,
                step_size=0.1)
        _logger.info(sensitivities)
        _, ratios = self.get_ratios_by_sensitive(sensitivities, pruned_flops,
                                                 eval_program)

        pruned_program = self._pruner.prune(
            train_program,
            self._scope,
            params,
            ratios,
            place=self._place,
            only_graph=False)
        pruned_val_program = None
        if eval_program is not None:
            pruned_val_program = self._pruner.prune(
                eval_program,
                self._scope,
                params,
                ratios,
                place=self._place,
                only_graph=True)
        self._iter += 1
        return pruned_program, pruned_val_program

    def _greedy_ratio_by_sensitive(self, sensitivities, topk=1):
        losses = {}
        percents = {}
        for param in sensitivities:
            losses[param] = sensitivities[param]['loss'][0]
            percents[param] = sensitivities[param]['pruned_percent'][0]
        topk_parms = sorted(losses, key=losses.__getitem__)[:topk]
        topk_percents = [percents[param] for param in topk_parms]
        return topk_parms, topk_percents

    def get_ratios_by_sensitive(self, sensitivities, pruned_flops,
                                eval_program):
        """
        Search a group of ratios for pruning target flops.

        Args:

          sensitivities(dict): The sensitivities used to generate a group of pruning ratios. The key of dict
                               is name of parameters to be pruned. The value of dict is a list of tuple with
                               format `(pruned_ratio, accuracy_loss)`.
          pruned_flops(float): The percent of FLOPS to be pruned.
          eval_program(Program): The program whose FLOPS is considered.

        Returns:

          dict: A group of ratios. The key of dict is name of parameters while the value is the ratio to be pruned.
        """

        min_loss = 0.
        max_loss = 0.
        # step 2: Find a group of ratios by binary searching.
        base_flops = flops(eval_program)
        ratios = None
        max_times = 20
        while min_loss < max_loss and max_times > 0:
            loss = (max_loss + min_loss) / 2
            _logger.info(
                '-----------Try pruned ratios while acc loss={}-----------'.
                format(loss))
            ratios = self.get_ratios_by_loss(sensitivities, loss)
            _logger.info('Pruned ratios={}'.format(
                [round(ratio, 3) for ratio in ratios.values()]))
            pruned_program = self._pruner.prune(
                eval_program,
                None,  # scope
                ratios.keys(),
                ratios.values(),
                None,  # place
                only_graph=True)
            pruned_ratio = 1 - (float(flops(pruned_program)) / base_flops)
            _logger.info('Pruned flops: {:.4f}'.format(pruned_ratio))

            # Check whether current ratios is enough
            if abs(pruned_ratio - pruned_flops) < 0.015:
                break
            if pruned_ratio > pruned_flops:
                max_loss = loss
            else:
                min_loss = loss
            max_times -= 1
        return ratios
