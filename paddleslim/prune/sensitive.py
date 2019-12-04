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

import sys
import os
import logging
import pickle
import numpy as np
import paddle.fluid as fluid
from ..core import GraphWrapper
from ..common import get_logger
from ..analysis import flops
from ..prune import Pruner

_logger = get_logger(__name__, level=logging.INFO)

__all__ = ["sensitivity", "flops_sensitivity"]


def sensitivity(program,
                place,
                param_names,
                eval_func,
                sensitivities_file=None,
                pruned_ratios=None):
    scope = fluid.global_scope()
    graph = GraphWrapper(program)
    sensitivities = _load_sensitivities(sensitivities_file)

    if pruned_ratios is None:
        pruned_ratios = np.arange(0.1, 1, step=0.1)

    for name in param_names:
        if name not in sensitivities:
            size = graph.var(name).shape()[0]
            sensitivities[name] = {
                'pruned_percent': [],
                'loss': [],
                'size': size
            }
    baseline = None
    for name in sensitivities:
        for ratio in pruned_ratios:
            if ratio in sensitivities[name]['pruned_percent']:
                _logger.debug('{}, {} has computed.'.format(name, ratio))
                continue
            if baseline is None:
                baseline = eval_func(graph.program)

            param_backup = {}
            pruner = Pruner()
            _logger.info("sensitive - param: {}; ratios: {}".format(name,
                                                                    ratio))
            pruned_program = pruner.prune(
                program=graph.program,
                scope=scope,
                params=[name],
                ratios=[ratio],
                place=place,
                lazy=True,
                only_graph=False,
                param_backup=param_backup)
            pruned_metric = eval_func(pruned_program)
            loss = (baseline - pruned_metric) / baseline
            _logger.info("pruned param: {}; {}; loss={}".format(name, ratio,
                                                                loss))

            for brother in pruner.pruned_list[0]:
                if brother in sensitivities:
                    sensitivities[name]['pruned_percent'].append(ratio)
                    sensitivities[name]['loss'].append(loss)

            _save_sensitivities(sensitivities, sensitivities_file)

            # restore pruned parameters
            for param_name in param_backup.keys():
                param_t = scope.find_var(param_name).get_tensor()
                param_t.set(param_backup[param_name], place)
    return sensitivities


def flops_sensitivity(program,
                      place,
                      param_names,
                      eval_func,
                      sensitivities_file=None,
                      pruned_flops_rate=0.1):

    assert (1.0 / len(param_names) > pruned_flops_rate)

    scope = fluid.global_scope()
    graph = GraphWrapper(program)
    sensitivities = _load_sensitivities(sensitivities_file)

    for name in param_names:
        if name not in sensitivities:
            size = graph.var(name).shape()[0]
            sensitivities[name] = {
                'pruned_percent': [],
                'loss': [],
                'size': size
            }
    base_flops = flops(program)
    target_pruned_flops = base_flops * pruned_flops_rate

    pruner = Pruner()
    baseline = None
    for name in sensitivities:

        pruned_program = pruner.prune(
            program=graph.program,
            scope=None,
            params=[name],
            ratios=[0.5],
            place=None,
            lazy=False,
            only_graph=True)
        param_flops = (base_flops - flops(pruned_program)) * 2
        channel_size = sensitivities[name]["size"]
        pruned_ratio = target_pruned_flops / float(param_flops)
        pruned_size = round(pruned_ratio * channel_size)
        pruned_ratio = 1 if pruned_size >= channel_size else pruned_ratio

        if len(sensitivities[name]["pruned_percent"]) > 0:
            _logger.debug('{} exist; pruned ratio: {}; excepted ratio: {}'.
                          format(name, sensitivities[name]["pruned_percent"][
                              0], pruned_ratio))
            continue
        if baseline is None:
            baseline = eval_func(graph.program)
        param_backup = {}
        pruner = Pruner()
        _logger.info("sensitive - param: {}; ratios: {}".format(name,
                                                                pruned_ratio))
        loss = 1
        if pruned_ratio < 1:
            pruned_program = pruner.prune(
                program=graph.program,
                scope=scope,
                params=[name],
                ratios=[pruned_ratio],
                place=place,
                lazy=True,
                only_graph=False,
                param_backup=param_backup)
            pruned_metric = eval_func(pruned_program)
            loss = (baseline - pruned_metric) / baseline
        _logger.info("pruned param: {}; {}; loss={}".format(name, pruned_ratio,
                                                            loss))
        sensitivities[name]['pruned_percent'].append(pruned_ratio)
        sensitivities[name]['loss'].append(loss)
        _save_sensitivities(sensitivities, sensitivities_file)

        # restore pruned parameters
        for param_name in param_backup.keys():
            param_t = scope.find_var(param_name).get_tensor()
            param_t.set(param_backup[param_name], place)
    return sensitivities


def _load_sensitivities(sensitivities_file):
    """
    Load sensitivities from file.
    """
    sensitivities = {}
    if sensitivities_file and os.path.exists(sensitivities_file):
        with open(sensitivities_file, 'rb') as f:
            if sys.version_info < (3, 0):
                sensitivities = pickle.load(f)
            else:
                sensitivities = pickle.load(f, encoding='bytes')

    for param in sensitivities:
        sensitivities[param]['pruned_percent'] = [
            round(p, 2) for p in sensitivities[param]['pruned_percent']
        ]
    return sensitivities


def _save_sensitivities(sensitivities, sensitivities_file):
    """
        Save sensitivities into file.
        """
    with open(sensitivities_file, 'wb') as f:
        pickle.dump(sensitivities, f)
