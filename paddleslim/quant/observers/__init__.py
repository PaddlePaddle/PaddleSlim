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

from .hist import HistObserver
from .kl import KLObserver
from .mse import MSEObserver
from .emd import EMDObserver
from .avg import AVGObserver
from .abs_max import AbsmaxObserver
from .mse_weight import MSEChannelWiseWeightObserver
from .abs_max_weight import AbsMaxChannelWiseWeightObserver

__all__ = [
    "HistObserver",
    "KLObserver",
    "MSEObserver",
    "EMDObserver",
    "AVGObserver",
    "MSEWeightObserver",
    "AbsmaxObserver",
    "MSEChannelWiseWeightObserver",
    "AbsMaxChannelWiseWeightObserver",
]
