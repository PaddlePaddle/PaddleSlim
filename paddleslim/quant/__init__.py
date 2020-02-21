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

import paddle.fluid as fluid
from ..common import get_logger

_logger = get_logger(__name__, level=logging.INFO)

try:
    fluid.require_version('1.7.0')
    from .quanter import quant_aware, quant_post, convert
except Exception as e:
    _logger.warning(
        "If you want to use training-aware and post-training quantization, "
        "please use Paddle >= 1.7.0 or develop version")

from .quant_embedding import quant_embedding
