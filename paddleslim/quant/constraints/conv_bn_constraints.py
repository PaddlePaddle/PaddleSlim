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

from .constraint import FusionConstraint
from ..nn import Conv2DBatchNormWrapper, QuantedConv2DBatchNorm

FUSED_LAYER = Conv2DBatchNormWrapper
QAT_FUSED_LAYER = QuantedConv2DBatchNorm


class FreezedConvBNConstraint(FusionConstraint):
    """ Simulate the folding of convolution and batch norm with a correction strategy.
    Reference to: Quantizing deep convolutional networks for efficient inference: A whitepaper
    Args:
        - freeze_bn_delay(int): After the 'freeze_bn_delay' steps training, switch from using batch
    statistics to long-term moving averages for batch normalization.

    Examples:
       .. code-block:: python

            import paddle
            from paddleslim.quant import SlimQuantConfig
            from paddleslim.quant import SlimQAT
            from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
            from paddle.vision.models import resnet18

            model = resnet18()
            quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
            q_config = SlimQuantConfig(activation=quanter, weight=quanter)
            q_config.add_constraints(FreezedConvBNConstraint(freeze_bn_delay=1))
            qat = SlimQAT(q_config)
            quant_model = qat.quantize(model, inplace=True, inputs=x)

    """

    def __init__(self, freeze_bn_delay=0):
        self._freeze_bn_delay = freeze_bn_delay

    def apply(self, model, graph, config):
        conv_bn_pairs = graph.find_conv_bn()
        for pair in conv_bn_pairs:
            pair = [node._layer for node in pair]
            self.fuse_ops(model, FUSED_LAYER, pair, config)
            config.add_qat_layer_mapping(FUSED_LAYER, QAT_FUSED_LAYER)

        def _set_freeze_bn_delay(layer):
            if isinstance(layer, FUSED_LAYER):
                setattr(layer, "_freeze_bn_delay", self._freeze_bn_delay)

        model.apply(_set_freeze_bn_delay)
