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

import numpy as np
import paddle
from .utils import get_ln_linear_info, find_parent_layer_and_sub_name
from .utils_layers import ShiftSmoothHelpLayer, WOBiasHelpLayer
__all__ = ['Smooth']


class Smooth():
    def __init__(
            self,
            model,
            model_config,
            alpha=0.5,
            smooth_all_linears=False,
            sample_function=None,
            search_function=None, ):
        '''
        Smooth is an updated version of SmoothQuant(https://arxiv.org/abs/2211.10438). 
        Compared with SmoothQuant, the piecewise smooth algorithm has the following updates:
        It supports the search functions to search best smooth scale.
        It supports the sample function to sample smooth scale.
        Currently, Smooth only supports linear layer and ColumnParallelLinear/RowParallelLinear layer.

        
        Args:
            model(paddle.nn.Layer, required): the model to be smoothed 
            model_config (dict, required): the config of model to be smoothed 
            alpha(float, optional): smoothing parameter. Default: 0.5
            smooth_all_linears(bool, optional): whether to smooth all linears. Default: False
            sample_function(function, optional): the function of sample to sample data. Default: None
            sample_start_step(int, optional): the step of sample data by using sample_function. Default: 0
            search_function(function, optional): the function of search smooth scale. Default: None
            
        Examples:
        .. code-block:: python
       
        from paddleslim.quant.advanced import Smooth
        smooth = Smooth(model)
        for data in dataloader():
            model(data)
            smooth.step += 1
        smooth.update_weight()
        '''
        self.model = model
        self.model_config = model_config
        self.fused_qkv = model_config.get("fused_qkv", True)
        self.linear_flag = model_config.get("linear_flag", 'linear')
        self.norm_flag = model_config.get("norm_flag", 'norm')
        self.parallel_ffn = model_config.get("parallel_ffn", False)
        self.skip_norm_list = model_config.get("skip_norm_list", [])
        self.skip_linear_list = model_config.get("skip_linear_list", [])

        self.alpha = alpha
        self.smooth_all_linears = smooth_all_linears
        self.sample_function = sample_function
        self.search_function = search_function

        self.model.eval()
        self.step = 0
        self.print_step = 1
        self.got_smooth_layers = False
        self.help_layers_ready = False
        self.layer_order = []
        self.scale_dict = {}
        self.smooth_scale_dict = {}
        self.sampled_inputs = {}
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

    def _get_smooth_layers(self):
        self.ln_linear_dict, self.linear_ln_dict = get_ln_linear_info(
            self.layer_order, self.norm_flag, self.linear_flag, self.fused_qkv,
            self.parallel_ffn, self.skip_norm_list)

        assert len(self.ln_linear_dict) > 0, 'No LN/Linear pair found'
        for key in self.ln_linear_dict:
            print('smooth pair LN {} : Linear {}'.format(
                key, self.ln_linear_dict[key]))
        if self.smooth_all_linears:
            if not self.help_layers_ready:
                rest_linears = [
                    l for l in self.layer_order
                    if self.linear_flag in l and l not in self.linear_ln_dict
                    and l not in self.skip_linear_list
                ]
                print('Preparing smooth layers', rest_linears)
                for cur_name, sub_layer in self.model.named_sublayers():
                    if sub_layer.full_name() in rest_linears:
                        new_layer = ShiftSmoothHelpLayer(sub_layer)
                        parent_layer, sub_name = find_parent_layer_and_sub_name(
                            self.model, cur_name)
                        setattr(parent_layer, sub_name, new_layer)
                        forward_pre_hook_handle = new_layer.register_forward_pre_hook(
                            self._forward_pre_hook)
                        self._forward_hook_list.append(forward_pre_hook_handle)

        self.got_smooth_layers = True

    def _forward_pre_hook(self, layer, input):
        '''
        when step 0, forward once and collect smooth layers.
        '''
        if self.step == 0 and layer.full_name() in self.layer_order:
            self.step += 1
        if self.step == 0:
            self.layer_order.append(layer.full_name())
            return input
        if self.step == 1:
            if not self.got_smooth_layers:
                self._get_smooth_layers()
        if self.step > 0:
            if layer.full_name() in self.linear_ln_dict.keys():
                self._sample_scale(input,
                                   self.linear_ln_dict[layer.full_name()])
            if type(layer) == ShiftSmoothHelpLayer:
                self._sample_scale(input, layer.full_name())

        return input

    def _sample_scale(self, input, ln_name):
        x = input[0] if type(input) == tuple else input
        x.stop_gradient = True
        x_abs_max = x.abs().max(axis=1, keepdim=True)
        x_abs_max = x_abs_max.max(axis=0)

        if ln_name not in self.scale_dict:
            self.sampled_inputs[ln_name] = x
            self.scale_dict[ln_name] = x_abs_max
        else:
            if self.sample_function is not None:
                self.sampled_inputs[ln_name] = self.sample_function.sample(
                    x, self.sampled_inputs[ln_name], ln_name)
            else:
                self.sampled_inputs[ln_name] = x
            tmp1 = paddle.concat([x_abs_max, self.scale_dict[ln_name]], axis=0)
            self.scale_dict[ln_name] = tmp1.max(axis=0, keepdim=True)

        # per step print once
        if self.print_step == self.step:
            print('[Smooth] Step [{}]: {}. abs_min: {}, abs_max: {}'.format(
                self.step, ln_name,
                float(self.scale_dict[ln_name].cast("float32").min()),
                float(self.scale_dict[ln_name].cast("float32").max())))
            if ln_name == list(self.linear_ln_dict.values())[-1]:
                self.print_step += 1

    def update_weight(self):

        for _, sub_layer in self.model.named_sublayers():
            layer_name = sub_layer.full_name()
            ln_name = None
            if layer_name in self.linear_ln_dict:
                ln_name = self.linear_ln_dict[layer_name]
            if type(sub_layer) == ShiftSmoothHelpLayer:
                ln_name = layer_name
            if ln_name is not None:
                act_abs_max = self.scale_dict[ln_name].cast("float32")
                sampled_input = self.sampled_inputs[ln_name].cast("float32")
                for param in sub_layer.parameters(include_sublayers=False):
                    if 'w_0' in param.name:
                        weight = param.cast("float32")
                        if self.search_function is not None:
                            s = self.search_function.search(
                                layer_name, sampled_input, act_abs_max, weight)
                        else:
                            w_abs_max = weight.abs().max(axis=-1, keepdim=True)
                            rw_abs_max = w_abs_max.reshape(act_abs_max.shape)
                            act_abs_max_np = act_abs_max.numpy()
                            weight_abs_max_np = rw_abs_max.numpy()
                            s = (
                                np.power(act_abs_max_np, self.alpha) / np.power(
                                    weight_abs_max_np, 1 - self.alpha)).clip(
                                        min=1e-5)
                            s = paddle.to_tensor(s, dtype="float32")

                        self.smooth_scale_dict[ln_name] = s.cast(param.dtype)
                        break

        # update linear weight
        for _, sub_layer in self.model.named_sublayers():
            layer_name = sub_layer.full_name()
            if layer_name in self.linear_ln_dict:
                for param in sub_layer.parameters(include_sublayers=False):
                    if 'w_0' in param.name:
                        ln_name = self.linear_ln_dict[layer_name]
                        print("[smooth] before linear [{}] weight, abs_max: {}".
                              format(param.name,
                                     float(param.cast("float32").abs().max())))
                        param_tmp = param * self.smooth_scale_dict[
                            ln_name].transpose(perm=[1, 0])
                        paddle.assign(param_tmp, output=param)
                        print("[smooth] after linear [{}] weight, abs_max: {}".
                              format(param.name,
                                     float(param_tmp.abs().max().cast(
                                         "float32"))))

        # update LN weight
        for cur_name, sub_layer in self.model.named_sublayers():
            layer_name = sub_layer.full_name()
            if layer_name in self.ln_linear_dict:
                s = self.smooth_scale_dict[layer_name].squeeze()
                for param in sub_layer.parameters(include_sublayers=False):
                    print("[smooth] before layer_norm {} weight, abs_max: {}".
                          format(param.name,
                                 float(param.abs().max().cast("float32"))))
                    param_tmp = param / s
                    paddle.assign(param_tmp, output=param)
                    print("[smooth] after layer_norm {} weight, abs_max: {}".
                          format(param.name,
                                 float(param_tmp.abs().max().cast("float32"))))
                if not hasattr(sub_layer, "bias") or sub_layer.bias is None:
                    parent_layer, _ = find_parent_layer_and_sub_name(
                        self.model, cur_name)
                    if type(parent_layer) == WOBiasHelpLayer:
                        param = parent_layer.bias
                        print(
                            "[smooth WOBiasHelpLayer] before layer_norm {} bias, abs_max: {}".
                            format(param.name,
                                   float(param.abs().max().cast("float32"))))
                        param_tmp = param / s
                        paddle.assign(param_tmp, output=param)
                        print(
                            "[smooth WOBiasHelpLayer] after layer_norm {} bias, abs_max: {}".
                            format(param.name,
                                   float(param_tmp.abs().max().cast(
                                       "float32"))))

        for _, sub_layer in self.model.named_sublayers():
            if type(sub_layer) == ShiftSmoothHelpLayer:
                layer_name = sub_layer.full_name()
                linear_name = sub_layer.layer.full_name()
                smooth_scale = self.smooth_scale_dict[layer_name]
                print(
                    "[smooth ShiftSmoothHelpLayer] param: {}, before weight, abs_max: {}".
                    format(linear_name,
                           float(sub_layer.weight.abs().max().cast("float32"))))
                sub_layer.convert_weight(smooth_weight=smooth_scale)
                print(
                    "[smooth ShiftSmoothHelpLayer] param: {}, after weight, abs_max: {}".
                    format(linear_name,
                           float(sub_layer.weight.abs().max().cast("float32"))))

        self._remove_hook()
        paddle.device.cuda.empty_cache()

    def _remove_hook(self):
        for hook in self._forward_hook_list:
            hook.remove()
        self._forward_hook_list = []
