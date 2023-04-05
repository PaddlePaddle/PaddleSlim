# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import abc
import paddle
import numpy as np
from paddle.quantization.base_quanter import BaseQuanter


class BaseFakeQuanterLayer(BaseQuanter):
    def __init__(
            self,
            quant_bits=8,
            sign=True,
            symmetric=True, ):
        super(BaseFakeQuanterLayer, self).__init__()
        self._quant_bits = quant_bits
        self._sign = sign
        self._symmetric = symmetric

        self._min = None
        self._max = None
        self._qmin = None
        self._qmax = None

        self._scale = None
        self._zero_point = None

    @property
    def qmin_qmax(self):
        """ Get the range of the integer."""
        if self._qmin is not None and self._qmax is not None:
            return self.qmin, self.qmax
        if self._sign:
            self.qmin = -2**(self.bit_length() - 1)
            self.qmax = 2**(self.bit_length() - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**self.bit_length()
        return self.qmin, self.qmax
