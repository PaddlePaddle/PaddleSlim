#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import re
import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D


class AdamW(F.optimizer.AdamOptimizer):
    """AdamW object for dygraph"""

    def __init__(self, *args, **kwargs):
        weight_decay = kwargs.pop('weight_decay', None)
        var_name_to_exclude = kwargs.pop(
            'var_name_to_exclude', '.*layer_norm_scale|.*layer_norm_bias|.*b_0')
        super(AdamW, self).__init__(*args, **kwargs)
        self.wd = weight_decay
        self.pat = re.compile(var_name_to_exclude)

    def apply_optimize(self, loss, startup_program, params_grads):
        super(AdamW, self).apply_optimize(loss, startup_program, params_grads)
        for p, g in params_grads:
            if not self.pat.match(p.name):
                with D.no_grad():
                    L.assign(p * (1. - self.wd * self.current_step_lr()), p)
