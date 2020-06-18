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

from __future__ import absolute_import

__all__ = []
try:
    from paddleslim import models
    from paddleslim import prune
    from paddleslim import nas
    from paddleslim import analysis
    from paddleslim import quant
    from paddleslim import pantheon
    __all__ += ['models', 'prune', 'nas', 'analysis', 'quant', 'pantheon']
except ImportError:
    print(
        "PaddlePaddle is not installed in your env. So you can not use some APIs."
    )

from paddleslim import dist
__all__ += ['dist']
