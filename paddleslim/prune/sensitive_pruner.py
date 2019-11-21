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

import logging
import copy
from scipy.optimize import leastsq
import numpy as np
import paddle.fluid as fluid
from ..common import get_logger
from ..analysis import sensitivity
from ..analysis import flops
from .pruner import Pruner

__all__ = ["SensitivePruner"]

_logger = get_logger(__name__, level=logging.INFO)


class SensitivePruner(object):
    def __init__(self, place, eval_func, scope=None):
        """
        Pruner used to prune parameters iteratively according to sensitivities of parameters in each step.
        Args:
            place(fluid.CUDAPlace | fluid.CPUPlace): The device place where program execute.
            eval_func(function): A callback function used to evaluate pruned program. The argument of this function is pruned program. And it return a score of given program.
            scope(fluid.scope): The scope used to execute program.
        """
        self._eval_func = eval_func
        self._iter = 0
        self._place = place
        self._scope = fluid.global_scope() if scope is None else scope
        self._pruner = Pruner()

    def prune(self, train_program, eval_program, params, pruned_flops):
        """
        Pruning parameters of training and evaluation network by sensitivities in current step.
        Args:
            train_program(fluid.Program): The training program to be pruned.
            eval_program(fluid.Program): The evaluation program to be pruned. And it is also used to calculate sensitivities of parameters.
            params(list<str>): The parameters to be pruned.
            pruned_flops(float): The ratio of FLOPS to be pruned in current step.
        Return:
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
        print sensitivities
        _, ratios = self._get_ratios_by_sensitive(sensitivities, pruned_flops,
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

    def _get_ratios_by_sensitive(self, sensitivities, pruned_flops,
                                 eval_program):
        """
        Search a group of ratios for pruning target flops.
        """

        def func(params, x):
            a, b, c, d = params
            return a * x * x * x + b * x * x + c * x + d

        def error(params, x, y):
            return func(params, x) - y

        def slove_coefficient(x, y):
            init_coefficient = [10, 10, 10, 10]
            coefficient, loss = leastsq(error, init_coefficient, args=(x, y))
            return coefficient

        min_loss = 0.
        max_loss = 0.

        # step 1: fit curve by sensitivities
        coefficients = {}
        for param in sensitivities:
            losses = np.array([0] * 5 + sensitivities[param]['loss'])
            precents = np.array([0] * 5 + sensitivities[param][
                'pruned_percent'])
            coefficients[param] = slove_coefficient(precents, losses)
            loss = np.max(losses)
            max_loss = np.max([max_loss, loss])

        # step 2: Find a group of ratios by binary searching.
        base_flops = flops(eval_program)
        ratios = []
        max_times = 20
        while min_loss < max_loss and max_times > 0:
            loss = (max_loss + min_loss) / 2
            _logger.info(
                '-----------Try pruned ratios while acc loss={}-----------'.
                format(loss))
            ratios = []
            # step 2.1: Get ratios according to current loss
            for param in sensitivities:
                coefficient = copy.deepcopy(coefficients[param])
                coefficient[-1] = coefficient[-1] - loss
                roots = np.roots(coefficient)
                for root in roots:
                    min_root = 1
                    if np.isreal(root) and root > 0 and root < 1:
                        selected_root = min(root.real, min_root)
                ratios.append(selected_root)
            _logger.info('Pruned ratios={}'.format(
                [round(ratio, 3) for ratio in ratios]))
            # step 2.2: Pruning by current ratios
            param_shape_backup = {}
            pruned_program = self._pruner.prune(
                eval_program,
                None,  # scope
                sensitivities.keys(),
                ratios,
                None,  # place
                only_graph=True)
            pruned_ratio = 1 - (float(flops(pruned_program)) / base_flops)
            _logger.info('Pruned flops: {:.4f}'.format(pruned_ratio))

            # step 2.3: Check whether current ratios is enough
            if abs(pruned_ratio - pruned_flops) < 0.015:
                break
            if pruned_ratio > pruned_flops:
                max_loss = loss
            else:
                min_loss = loss
            max_times -= 1
        return sensitivities.keys(), ratios
