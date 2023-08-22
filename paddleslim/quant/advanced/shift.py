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

from re import sub
import numpy as np
import paddle
from .utils import get_ln_linear_info, find_parent_layer_and_sub_name
from .utils_layers import ShiftSmoothHelpLayer, WOBiasHelpLayer

__all__ = ['Shift']


class Shift():
    def __init__(self,
                 model,
                 model_config,
                 shift_all_linears=False,
                 sample_function=None):
        '''
        Shift is the implementation of Outlier Suppression+(https://arxiv.org/abs/2304.09145). 
        Currently, Shift only supports linear layer and ColumnParallelLinear/RowParallelLinear layer.

        Args:
            model(paddle.nn.Layer, required): the model to be shifted 
            model_config (dict, required): the config of model to be shifted 
            shift_all_linears (bool, optional): whether to shift all linear layers 
            sample_function (function, optional): the function to sample data

        Examples:
        .. code-block:: python
       
        from paddleslim.quant.advanced import Shift
        shift = Shift(model)
        for data in dataloader():
            model(data)
            shift.step += 1
        shift.update_weight()
        '''

        self.model = model
        self.model_config = model_config
        self.fused_qkv = model_config.get("fused_qkv", True)
        self.linear_flag = model_config.get("linear_flag", 'linear')
        self.norm_flag = model_config.get("norm_flag", 'norm')
        self.parallel_ffn = model_config.get("parallel_ffn", False)
        self.skip_norm_list = model_config.get("skip_norm_list", [])
        self.skip_linear_list = model_config.get("skip_linear_list", [])
        self.shift_all_linears = shift_all_linears
        self.sample_function = sample_function

        self.layer_order = []
        self.zero_point_dict = {}
        self.smooth_scale_dict = {}

        self.model.eval()
        self.step = 0
        self.print_step = 1
        self.got_shift_layers = False
        self.help_layers_ready = False
        self._apply_hook()

    def _apply_hook(self):
        self._forward_hook_list = []
        for _, sub_layer in self.model.named_sublayers():
            if self.norm_flag in sub_layer.full_name(
            ) or self.linear_flag in sub_layer.full_name():
                forward_pre_hook_handle = sub_layer.register_forward_pre_hook(
                    self._forward_pre_hook)
                self._forward_hook_list.append(forward_pre_hook_handle)
            if type(sub_layer) == ShiftSmoothHelpLayer:
                self.help_layers_ready = True
                forward_pre_hook_handle = sub_layer.register_forward_pre_hook(
                    self._forward_pre_hook)
                self._forward_hook_list.append(forward_pre_hook_handle)

    def _get_shift_layers(self):
        self.ln_linear_dict, self.linear_ln_dict = get_ln_linear_info(
            self.layer_order, self.norm_flag, self.linear_flag, self.fused_qkv,
            self.parallel_ffn, self.skip_norm_list)

        assert len(self.ln_linear_dict) > 0, 'No LN/Linear pair found'
        for key in self.ln_linear_dict:
            print('shift pair LN {} : Linear {}'.format(
                key, self.ln_linear_dict[key]))
        if self.shift_all_linears:
            if not self.help_layers_ready:
                rest_linears = [
                    l for l in self.layer_order
                    if self.linear_flag in l and l not in self.linear_ln_dict
                    and l not in self.skip_linear_list
                ]
                print('Preparing shift layers', rest_linears)
                for cur_name, sub_layer in self.model.named_sublayers():
                    if sub_layer.full_name() in rest_linears:
                        new_layer = ShiftSmoothHelpLayer(sub_layer)
                        parent_layer, sub_name = find_parent_layer_and_sub_name(
                            self.model, cur_name)
                        setattr(parent_layer, sub_name, new_layer)
                        forward_pre_hook_handle = new_layer.register_forward_pre_hook(
                            self._forward_pre_hook)
                        self._forward_hook_list.append(forward_pre_hook_handle)

        self.got_shift_layers = True

    def _forward_pre_hook(self, layer, input):
        '''
        when step 0, forward once and collect shift layers.
        when step >1, sample scale.
        '''
        if self.step == 0 and layer.full_name() in self.layer_order:
            self.step += 1
        if self.step == 0:
            self.layer_order.append(layer.full_name())
            return input
        if self.step == 1:
            if not self.got_shift_layers:
                self._get_shift_layers()
        if self.step > 0:
            if layer.full_name() in self.linear_ln_dict.keys():
                self._sample_zero_point(input,
                                        self.linear_ln_dict[layer.full_name()])
            if type(layer) == ShiftSmoothHelpLayer:
                self._sample_zero_point(input, layer.full_name())

        return input

    def _sample_zero_point(self, input, ln_name):
        x = input[0] if type(input) == tuple else input
        x.stop_gradient = True

        zero_point = x.mean(axis=(0, 1)) if len(x.shape) > 2 else x.mean(axis=1)
        _min = x.min(axis=(0, 1)) if len(x.shape) > 2 else x.min(axis=1)
        _max = x.max(axis=(0, 1)) if len(x.shape) > 2 else x.max(axis=1)

        if ln_name not in self.zero_point_dict:

            if self.sample_function is None:
                self.zero_point_dict[ln_name] = (_min + _max) / 2
            else:
                self.zero_point_dict[ln_name] = zero_point

        else:
            if self.sample_function is not None:
                self.zero_point_dict[ln_name] = self.sample_function.sample(
                    zero_point, self.zero_point_dict[ln_name], ln_name)
            else:
                cur_zero_point = (_min + _max) / 2
                self.zero_point_dict[ln_name] = (
                    self.zero_point_dict[ln_name] + cur_zero_point) / 2

        # per step print once
        if self.print_step == self.step:
            print('[shift] Step [{}]: {}. zero_point min: {}, max: {}'.format(
                self.step, ln_name,
                round(float(self.zero_point_dict[ln_name].min()), 5),
                round(float(self.zero_point_dict[ln_name].max()), 5)))
            if ln_name == list(self.linear_ln_dict.values())[-1]:
                self.print_step += 1

    def update_weight(self):
        '''
        update weight of smooth layers.
        firstly compute s and update linear's weight,
        then update LN's weight by corresponding linear and s
        '''
        # update linear weight
        for _, sub_layer in self.model.named_sublayers():
            layer_name = sub_layer.full_name()
            if layer_name in self.linear_ln_dict:
                ln_name = self.linear_ln_dict[layer_name]
                shift_bias = None
                for param in sub_layer.parameters(include_sublayers=False):
                    if 'w_0' in param.name:
                        zero_point = self.zero_point_dict[
                            ln_name].squeeze().cast(param.dtype)
                        shift_bias = paddle.matmul(zero_point, param)
                        print("[shift] param: {}, zero_point min: {}, max: {}".
                              format(param.name,
                                     float(zero_point.cast("float32").min()),
                                     float(zero_point.cast("float32").max())))
                        break

                if not hasattr(sub_layer, "bias") or sub_layer.bias is None:
                    sub_layer.bias = paddle.create_parameter(
                        shape=shift_bias.shape,
                        dtype=sub_layer.weight.dtype,
                        default_initializer=paddle.nn.initializer.Constant(0.0),
                        is_bias=True, )
                for param in sub_layer.parameters(include_sublayers=False):
                    if 'b_0' in param.name:
                        shift_bias = shift_bias + param
                        paddle.assign(
                            shift_bias.cast(param.dtype), output=param)
                        print("[shift] update linear bias: {}.".format(
                            param.name))
                        break

        # update LN weight
        for cur_name, sub_layer in self.model.named_sublayers():
            layer_name = sub_layer.full_name()
            if layer_name in self.ln_linear_dict:
                if not hasattr(sub_layer, "bias") or sub_layer.bias is None:
                    help_layer = WOBiasHelpLayer(sub_layer)
                    parent_layer, sub_name = find_parent_layer_and_sub_name(
                        self.model, cur_name)
                    setattr(parent_layer, sub_name, help_layer)
                    sub_layer = help_layer

                for param in sub_layer.parameters(include_sublayers=False):
                    if "b_0" in param.name:
                        zero_point = self.zero_point_dict[layer_name].squeeze()
                        param_tmp = param - zero_point
                        paddle.assign(param_tmp.cast(param.dtype), output=param)
                        print("[shift] update layer_norm bias {}.".format(
                            param.name))
                        break

        # update ShiftSmoothRowParallelLinear weight
        for _, sub_layer in self.model.named_sublayers():
            if type(sub_layer) == ShiftSmoothHelpLayer:
                layer_name = sub_layer.full_name()
                linear_name = sub_layer.layer.full_name()
                zero_point = self.zero_point_dict[layer_name].squeeze()
                print(
                    "[shift ShiftSmoothHelpLayer] before param: {}, shift_bias min: {}, max: {}".
                    format(linear_name,
                           float(sub_layer.shift_bias.cast("float32").min()),
                           float(sub_layer.shift_bias.max().cast("float32"))))
                sub_layer.convert_weight(shift_bias=zero_point)

                print(
                    "[shift ShiftSmoothHelpLayer] after param: {}, shift_bias min: {}, max: {}".
                    format(linear_name,
                           float(sub_layer.shift_bias.cast("float32").min()),
                           float(sub_layer.shift_bias.max().cast("float32"))))

        self._remove_hook()
        paddle.device.cuda.empty_cache()

    def _remove_hook(self):
        for hook in self._forward_hook_list:
            hook.remove()
