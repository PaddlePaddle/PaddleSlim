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
from ..prune import Pruner

_logger = get_logger(__name__, level=logging.INFO)

__all__ = ["sensitivity"]


def sensitivity(program,
                place,
                param_names,
                eval_func,
                sensitivities_file=None,
                step_size=0.2,
                max_pruned_times=None):
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
    baseline = None
    for name in sensitivities:
        ratio = step_size
        pruned_times = 0
        while ratio < 1:
            if max_pruned_times is not None and pruned_times >= max_pruned_times:
                break
            ratio = round(ratio, 2)
            if ratio in sensitivities[name]['pruned_percent']:
                _logger.debug('{}, {} has computed.'.format(name, ratio))
                ratio += step_size
                pruned_times += 1
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
            sensitivities[name]['pruned_percent'].append(ratio)
            sensitivities[name]['loss'].append(loss)
            _save_sensitivities(sensitivities, sensitivities_file)

            # restore pruned parameters
            for param_name in param_backup.keys():
                param_t = scope.find_var(param_name).get_tensor()
                param_t.set(param_backup[param_name], place)
            ratio += step_size
            pruned_times += 1
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
