#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import os
import sys
import logging
import paddle.fluid as fluid
from ..log_helper import get_logger

__all__ = ['RLBaseController']

_logger = get_logger(__name__, level=logging.INFO)


class RLBaseController(object):
    """ Base Controller for reforcement learning"""

    def next_tokens(self, *args, **kwargs):
        raise NotImplementedError('Abstract method.')

    def update(self, *args, **kwargs):
        raise NotImplementedError('Abstract method.')

    def _save_controller(self, program, output_dir):
        fluid.save(program, output_dir)

    def _load_controller(self, program, load_dir):
        fluid.load(program, load_dir)
