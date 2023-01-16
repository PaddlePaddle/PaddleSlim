# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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

import copy
import logging

import paddle
import paddle.nn as nn

from paddle.quantization import (
    PTQConfig,
    ImperativePTQ,
    AbsmaxQuantizer,
    HistQuantizer,
    KLQuantizer,
    PerChannelAbsmaxQuantizer,
    SUPPORT_ACT_QUANTIZERS,
    SUPPORT_WT_QUANTIZERS, )
from ...common import get_logger

_logger = get_logger(__name__, level=logging.INFO)

__all__ = [
    'PTQ',
    'AbsmaxQuantizer',
    'HistQuantizer',
    'KLQuantizer',
    'PerChannelAbsmaxQuantizer',
]


class PTQ(object):
    """
    Static post training quantization.
    """

    def __init__(self,
                 activation_quantizer='KLQuantizer',
                 weight_quantizer='PerChannelAbsmaxQuantizer',
                 **kwargs):
        """
        Args:
            activation_quantizer(Quantizer): The quantizer method for activation.
                Can be set to `KLQuantizer`/`HistQuantizer`/`AbsmaxQuantizer`.
                Default: KLQuantizer.
            weight_quantizer(Quantizer): The quantizer method for weight.
                Can be set to `AbsmaxQuantizer`/`PerChannelAbsmaxQuantizer`.
                Default: PerChannelAbsmaxQuantizer.
        """
        print("activation_quantizer", activation_quantizer)
        activation_quantizer = eval(activation_quantizer)(**kwargs)
        weight_quantizer = eval(weight_quantizer)()
        assert isinstance(activation_quantizer, tuple(SUPPORT_ACT_QUANTIZERS))
        assert isinstance(weight_quantizer, tuple(SUPPORT_WT_QUANTIZERS))

        quant_config = PTQConfig(
            activation_quantizer=activation_quantizer,
            weight_quantizer=weight_quantizer)

        self.ptq = ImperativePTQ(quant_config=quant_config)

    def quantize(self, model, inplace=False, fuse=False, fuse_list=None):
        """
        Quantize the input model.

        Args:
            model(paddle.nn.Layer): The model to be quantized.
            inplace(bool): Whether apply quantization to the input model.
                           Default: False.
            fuse(bool): Whether to fuse layers.
                        Default: False.
            fuse_list(list): The layers' names to be fused. For example,
                "fuse_list = [["conv1", "bn1"], ["conv2", "bn2"]]".
                The conv2d and bn layers will be fused automatically
                if "fuse" was set as True but "fuse_list" was None.
                Default: None.
        Returns:
            quantized_model(paddle.nn.Layer): The quantized model.
        """
        assert isinstance(model, paddle.nn.Layer), \
            "The model must be the instance of paddle.nn.Layer."

        if fuse == True:
            if fuse_list is None:
                fuse_list = self.find_conv_bn_names(model)
            _logger.info('The layers to be fused:')
            for i in fuse_list:
                _logger.info(i)

        return self.ptq.quantize(
            model=model, inplace=inplace, fuse=fuse, fuse_list=fuse_list)

    def find_conv_bn_names(self, model):
        """
        Find the connected conv2d and bn layers of model.
       
        Args:
            model(paddle.nn.Layer): The model to be fuseed.
       
        Returns:
            fuse_list(list): The conv and bn layers to be fused.
        """

        last_layer = None
        fuse_list = []
        for name, layer in model.named_sublayers():
            if isinstance(last_layer, nn.Conv2D) and isinstance(layer,
                                                                nn.BatchNorm2D):
                fuse_list.append([last_name, name])
            last_name = name
            last_layer = layer

        return fuse_list

    def save_quantized_model(self, model, path, input_spec=None, **kwargs):
        """
        Save the quantized inference model.

        Args:
            model (Layer): The model to be saved.
            path (str): The path prefix to save model. The format is 
                ``dirname/file_prefix`` or ``file_prefix``.
            input_spec (list[InputSpec|Tensor], optional): Describes the input
                of the saved model's forward method, which can be described by
                InputSpec or example Tensor. If None, all input variables of 
                the original Layer's forward method would be the inputs of
                the saved model. Default: None.
            kwargs (dict, optional): Other save configuration options for compatibility.
        Returns:
            None
        """
        assert isinstance(model, paddle.nn.Layer), \
            "The model must be the instance of paddle.nn.Layer."

        training = model.training
        if training:
            model.eval()

        self.ptq.save_quantized_model(
            model=model, path=path, input_spec=input_spec, **kwargs)

        if training:
            model.train()
