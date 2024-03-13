# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.'

from . import mobilenet_v1
from . import mobilenet_v2

from .mobilenet_v1 import MobileNetV1_x0_25, MobileNetV1_x0_5, MobileNetV1_x0_75, MobileNetV1
from .mobilenet_v2 import MobileNetV2_x0_25, MobileNetV2_x0_5, MobileNetV2_x0_75, MobileNetV2, MobileNetV2_x1_5, MobileNetV2_x2_0

__all__ = mobilenet_v1.__all__ + mobilenet_v2.__all__