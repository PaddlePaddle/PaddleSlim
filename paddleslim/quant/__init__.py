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

import platform
import logging

import paddle
from ..common import get_logger

_logger = get_logger(__name__, level=logging.INFO)
min_paddle_version = '2.3.0'
try:
    paddle.utils.require_version(min_paddle_version)
    from .quanter import quant_aware, convert, quant_post_static, quant_post_dynamic
    from .quanter import quant_post, quant_post_only_weight
    from .quant_aware_with_infermodel import quant_aware_with_infermodel, export_quant_infermodel
    from .reconstruction_quantization import quant_recon_static
    if platform.system().lower() == 'linux':
        from .post_quant_hpo import quant_post_hpo
    else:
        _logger.warning(
            "post-quant-hpo is not support in system other than linux")

except Exception as e:
    _logger.warning(e)
    _logger.warning(
        "If you want to use training-aware and post-training quantization, "
        "please use Paddle >= {} or develop version".format(min_paddle_version))

from .quant_embedding import quant_embedding
from . import nn
from .qat import SlimQAT
from .config import SlimQuantConfig
