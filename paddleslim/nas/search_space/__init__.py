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

import mobilenetv2
from .mobilenetv2 import *
import mobilenetv2_block
from .mobilenetv2_block import *
import mobilenetv1
from .mobilenetv1 import *
import resnet
from .resnet import *
import search_space_registry
from search_space_registry import *
import search_space_factory
from search_space_factory import *
import search_space_base
from search_space_base import *

__all__ = []
__all__ += mobilenetv2.__all__
__all__ += search_space_registry.__all__
__all__ += search_space_factory.__all__
__all__ += search_space_base.__all__
