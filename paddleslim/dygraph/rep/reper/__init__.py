# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
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

from . import diversebranchblock
from . import acblock
from . import repvgg
from . import slimrep
from . import base

from .diversebranchblock import DiverseBranchBlock
from .acblock import ACBlock
from .repvgg import RepVGGBlock
from .slimrep import SlimRepBlock

__all__ = []
__all__ += diversebranchblock.__all__
__all__ += acblock.__all__
__all__ += repvgg.__all__
__all__ += slimrep.__all__
