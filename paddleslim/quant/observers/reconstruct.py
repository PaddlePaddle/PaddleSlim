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
import time
import logging
import copy
from tqdm import tqdm
import paddle
from paddle.quantization import PTQ
from paddle.quantization.base_quanter import BaseQuanter
from paddle.nn.quant.format import (
    ConvertibleQuantedLayer,
    LinearQuanterDequanter, )
from ...common import get_logger
from .reconstruct_weight import ReconstructWeightObserverLayer
from .reconstruct_act import ReconstructActObserverLayer

_logger = get_logger(
    __name__,
    logging.INFO,
    fmt='%(asctime)s-%(levelname)s: %(message)s', )

GAMMA = -0.1
ZETA = 1.1


class ReconstructPTQ(PTQ):
    """ Utilizing reconstruction quantization method to quantize the FP32 model, and it uses calibrate data to get the quantization information for all quantized variables.
    Args:
        model(paddle.nn.Layer): The FP32 model to be quantized.
        config(paddle.quantization.QuantConfig): The quant configuration.
        data_loader(Python Generator, Paddle.io.DataLoader): The
            Generator or Dataloader provides calibrate data, and it could
            return a batch every time.
        epochs(int): The number of epochs in the reconstruction proces. Default is 10.
        batch_nums (int): Total number of minibatchs used to calibrate quantized variables. It will cover the batch_nums in ReconstructActObserver and ReconstructWeightObserver. Default is 10.
        lr(float): The learning rate. Default is 0.1.
    """

    def __init__(self,
                 model,
                 config,
                 data_loader,
                 epochs=10,
                 batch_nums=10,
                 lr=0.1,
                 recon_level='layer-wise'):
        super(ReconstructPTQ, self).__init__(config)
        assert recon_level in [
            'layer-wise', 'region-wise'
        ], "recon_level must be one of the ['layer-wise', 'region-wise'], but received: {}".format(
            recon_level)
        self.origin_model = model
        self._data_loader = data_loader
        self._batch_nums = batch_nums
        self._epochs = epochs
        self._lr = lr
        self._recon_level = recon_level

        self.all_quant_layer_outputs = {}
        self.all_origin_layer_outputs = {}
        self._qat_layer_mapping = copy.deepcopy(config._qat_layer_mapping)
        self._qat_layer_mapping.pop(paddle.nn.quant.Stub)

    def init_ptq(self):
        """ Create a model for reconstruct ptq.
        """
        self.origin_model.eval()
        self.quant_model = self.quantize(self.origin_model, inplace=False)
        self._set_batch_nums()

        # apply hook
        if self._recon_level == 'layer-wise':
            for layer in self.quant_model.sublayers():
                if isinstance(layer, tuple(self._qat_layer_mapping.values())):
                    layer.register_forward_post_hook(
                        self._quant_forward_post_hook)
            for layer in self.origin_model.sublayers():
                if isinstance(layer, tuple(self._qat_layer_mapping.keys())):
                    layer.register_forward_post_hook(
                        self._origin_forward_post_hook)

        elif self._recon_level == 'region-wise':
            for block in self.quant_model.children():
                block.register_forward_post_hook(self._quant_forward_post_hook)
            for block in self.origin_model.children():
                block.register_forward_post_hook(self._origin_forward_post_hook)

    def _quant_forward_post_hook(self, layer, inputs, outputs):
        layer_name = layer.full_name().split("quanted_")[-1]
        self.all_quant_layer_outputs[layer_name] = outputs
        return outputs

    def _origin_forward_post_hook(self, layer, inputs, outputs):
        layer_name = layer.full_name()
        self.all_origin_layer_outputs[layer_name] = outputs
        return outputs

    def _set_batch_nums(self):
        for layer in self.quant_model.sublayers():
            if isinstance(layer, tuple(self._qat_layer_mapping.values())):
                if isinstance(layer.weight_quanter,
                              ReconstructWeightObserverLayer):
                    layer.weight_quanter.set_batch_nums(self._batch_nums)
                if isinstance(layer.activation_quanter,
                              ReconstructActObserverLayer):
                    layer.activation_quanter.set_batch_nums(self._batch_nums)

    def _get_layers(self):
        quant_model_layers = {}
        if self._recon_level == 'layer-wise':
            for layer in self.quant_model.sublayers():
                if isinstance(layer, tuple(self._qat_layer_mapping.values())):
                    layer_name = layer.full_name().split("quanted_")[-1]
                    quant_model_layers[layer_name] = []
                    quant_model_layers[layer_name].append(layer)

        elif self._recon_level == 'region-wise':
            for block in self.quant_model.children():
                quant_model_layers[block.full_name()] = []
                for layer in block.sublayers():
                    if isinstance(layer,
                                  tuple(self._qat_layer_mapping.values())):
                        quant_model_layers[block.full_name()].append(layer)
        return quant_model_layers

    def run(self):
        """ Use the calibrate data to calculate the forward-stage. Based on the sample data, utilizing reconstruction quantization method to update quanted model.

        Return: The converted model
        """
        self.origin_model.eval()
        self.quant_model.eval()
        with tqdm(
                total=self._batch_nums,
                bar_format=
                'Sampling stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
                ncols=80, ) as t:
            for batch_id, data in enumerate(self._data_loader()):
                # data (dict)
                self.quant_model(**data)
                t.update()
                if batch_id + 1 == self._batch_nums:
                    break

        quant_model_layers = self._get_layers()

        _logger.info("Begin updating quant model")
        for name, layers in quant_model_layers.items():
            _logger.info(f'Current layer: {name}')
            if len(layers) == 0:
                _logger.info(
                    'The layer does not have a quantifiable weight, skiping it.'
                )
                continue
            alphas = [layer.weight_quanter.alpha for layer in layers]
            opt = paddle.optimizer.Adam(
                learning_rate=self._lr, parameters=alphas)
            for epoch in range(self._epochs):
                for batch_id, data in enumerate(self._data_loader()):
                    # data (dict)
                    start_time = time.time()
                    self.origin_model(**data)
                    self.quant_model(**data)

                    round_loss = 0.0
                    for layer in layers:
                        h_alpha = layer.weight_quanter.compute_soft_rounding()
                        round_loss += self._round_loss(h_alpha)
                    recon_loss = self._recon_loss(name)
                    total_loss = round_loss + recon_loss
                    total_loss.backward()

                    opt.step()
                    opt.clear_grad()
                    cur_time = time.time()

                    _logger.info(
                        "Epoch {:d}, Iter {:d}, lr {}, total_loss {:.5f}, recon_loss {:.5f}, round_loss {:.5f}, time {:.5f}s"
                        .format(epoch, batch_id, self._lr,
                                total_loss.numpy()[0],
                                recon_loss.numpy()[0],
                                round_loss.numpy()[0], cur_time - start_time), )

                    if batch_id + 1 == self._batch_nums:
                        break
        _logger.info("Update done!")
        return self.quant_model

    def _round_loss(self, h_v):
        return paddle.sum(-paddle.pow(paddle.abs(2 * h_v - 1), 3) + 1)

    def _recon_loss(self, name):
        return paddle.nn.functional.mse_loss(
            self.all_quant_layer_outputs[name],
            self.all_origin_layer_outputs[name], )

    def _calculate_final_alpha(self, model):
        h_alpha_dict = {}
        for layer in model.sublayers():
            if isinstance(layer, tuple(self._qat_layer_mapping.values())):
                h_alpha = layer.weight_quanter.compute_soft_rounding()
                quant_weight = self._quant(layer.weight,
                                           layer.weight_quanter.scales(),
                                           layer.weight_quanter._qmax)
                adaround_weight = paddle.floor(quant_weight) + h_alpha
                new_alpha = adaround_weight - paddle.round(quant_weight)
                h_alpha_dict[layer.weight.name] = new_alpha
        return h_alpha_dict

    def _quant(self, x, scale, qmax):
        s = scale / qmax
        quant_x = x / s
        return quant_x

    def convert(self, model, inplace=True):
        """ Convert the quantization model to onnx style. And the converted
        model can be saved as inference model by calling paddle.jit.save.
        Args:
            model(Layer) - The quantized model to be covnerted.
            inplace(bool) - Whether to modify the model in-place.

        Return: The converted model
        """
        _model = model if inplace else copy.deepcopy(self.quant_model)
        h_alpha_dict = self._calculate_final_alpha(_model)

        replaced = {}
        for name, child in _model.named_children():
            quant_dequant = None
            if isinstance(child, ConvertibleQuantedLayer):
                child._convert()
                adaround_weight = child.weight + h_alpha_dict[child.weight.name]
                child.weight.set_value(adaround_weight)
            elif isinstance(child, BaseQuanter):
                quant_dequant = LinearQuanterDequanter.from_quanter(child)
            else:
                self.convert(child, inplace=True)
            if quant_dequant is not None:
                replaced[name] = quant_dequant
        for key, value in replaced.items():
            _model._sub_layers[key] = value
        return _model
