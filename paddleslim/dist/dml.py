# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
support_pd = False
support_torch = False
import torch.nn as nn
from .torch_dml import TORCH_DML
support_torch = True

try:
    import paddle.fluid as fluid
    from .pd_dml import PD_DML
    support_pd = True
except ImportError:
    print("")


def DML(model):
    """
    
    """
    if support_torch and isinstance(model, nn.Module):
        return TORCH_DML(model)
    elif support_pd and isinstance(model, fluid.dygraph.Layer):
        return PDDML(model)
    else:
        print("Please install paddlepaddle or pytorch.")
