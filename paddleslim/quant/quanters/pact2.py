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

import paddle
import numpy as np
from paddle.utils import unique_name
from paddle.quantization.factory import QuanterFactory
from paddle.quantization.base_quanter import BaseQuanter

PACT2_RANGE_INIT = 8.0


class PACT2Quanter(QuanterFactory):
    def __init__(self, quanter, update_range=True, clip_range=None):
        super(PACT2Quanter, self).__init__(
            quanter=quanter, update_range=update_range, clip_range=clip_range)

    def _get_class(self):
        return PACT2QuanterLayer


class PACT2QuanterLayer(BaseQuanter):
    def __init__(self, layer, quanter, update_range=True, clip_range=None):
        super(PACT2QuanterLayer, self).__init__()

        self.quanter = quanter(layer)
        self._update_range = update_range

        clip_init = max(abs(np.array(clip_range))) if (
            clip_range is not None) else PACT2_RANGE_INIT
        clip_init2 = np.power(2.0, np.ceil(np.log2(clip_init)))

        value_prefix = ("{}.pact2_range".format(layer.full_name()))
        name = unique_name.generate(value_prefix)

        clip_attr = paddle.ParamAttr(
            name=name,
            initializer=paddle.nn.initializer.Assign([-clip_init2, clip_init2]),
            trainable=False)

        self.clip_value = self.create_parameter(
            shape=[2], attr=clip_attr, dtype='float32')

        self._eps = 2.0**(-16.0)
        self._range_update_factor_min = 0.001
        self._current_iters = 0
        self._sign = True

    def forward(self, activation):
        self._current_iters += 1
        if self._update_range:
            with paddle.no_grad():
                self._update_clip_range(activation)
        clips = self._get_clips_value()
        act = paddle.clip(activation, clips[0], clips[1])
        return self.quanter(act)

    def _update_clip_range(self, act):
        clip_min, clip_max = self.clip_value[0], self.clip_value[1]

        # TODO: realize different extrema computation modes
        min_value = act.min()
        max_value = act.max()

        # exponential moving average update
        update_factor = 1.0 / self._current_iters
        update_factor = max(update_factor, self._range_update_factor_min)
        clip_max = clip_max * (1 - update_factor) + max_value * update_factor
        clip_min = clip_min * (1 - update_factor) + min_value * update_factor

        self.clip_value[0].set_value(clip_min)
        self.clip_value[1].set_value(clip_max)

    def _get_clips_value(self):
        if self._sign:
            clip_max = paddle.max(paddle.abs(self.clip_values))
        else:
            clip_max = paddle.abs(self.clip_values[1])
        clip_max = paddle.clip(clip_max, min=self._eps)
        clip_max2 = clip_max + (paddle.pow(
            2, paddle.ceil(paddle.log2(clip_max))) - clip_max).detach()
        clip_min2 = (-clip_max2 if self._sign else clip_max2 * 0.0)
        return (clip_min2, clip_max2)

    def bit_length(self):
        """ Return the bit length of quantized data.
        """
        return self.quanter.bit_length()

    def quant_axis(self):
        """ Return quantization axis.
        """
        return self.quanter.quant_axis()

    def scales(self):
        """ Return output scales.
        """
        return self.quanter.scales()

    def zero_points(self):
        """ Return output zero points.
        """
        return self.quanter.zero_points()
