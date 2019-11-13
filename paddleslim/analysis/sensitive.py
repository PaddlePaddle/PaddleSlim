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

import numpy as np
from ..core import GraphWrapper

__all__ = ["sensitivity"]


def sensitivity(program,
                scope,
                param_names,
                eval_func,
                sensitivities_file=None):

    graph = GraphWrapper(program)
    if sensitivities_file is not None:
        assert os.path.exsits(sensitivities_file)

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
        while ratio < 1:
            ratio = round(ratio, 2)
            if ratio in sensitivities[param]['pruned_percent']:
                _logger.debug('{}, {} has computed.'.format(name, ratio))
                ratio += step_size
                continue
            if baseline is None:
                baseline = _eval_func(grpah.program, scope)

            param_backup = {}
            pruner = Pruner()
            pruned_program = pruner.prune(
                program=graph.program,
                scope=scope,
                params=[name],
                ratios=[ratio],
                place=place,
                lazy=True,
                only_graph=False,
                param_backup=param_backup)
            pruned_metric = _eval_func(pruned_program)
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
    self._format_sensitivities(sensitivities)
    return sensitivities
