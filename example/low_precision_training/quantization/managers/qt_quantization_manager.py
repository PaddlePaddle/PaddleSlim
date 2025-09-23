import paddle
import paddle.nn as nn
import abc
import types
import copy
from addict import Dict

from .base_quantization_manager import BaseQuantizationManager
from . import register_quantization_manager
from ..layers import QTLinear, QTConv2d

@register_quantization_manager('qt_quantization_manager')
class QTQuantizationManager(BaseQuantizationManager):
    def __init__(self, qconfig):
        super(QTQuantizationManager, self).__init__(qconfig)
        self.qconfig = qconfig

    def enable_quantization(self, model):
        for m in model.sublayers():
            if isinstance(m, (QTLinear, QTConv2d)):
                m.enabled = True

    def disable_quantization(self, model):
        for m in model.sublayers():
            if isinstance(m, (QTLinear, QTConv2d)):
                m.enabled = False

    def prepare(self, model, inplace=True):
        if not inplace:
            model = copy.deepcopy(model)

        # convert Conv2d -> QTConv2d, etc
        self.convert_layers(model)

        # set bitwidth information for each module
        self.parse_quantization_params(model)

        model.enable_quantization = types.MethodType(self.enable_quantization, model)
        model.disable_quantization = types.MethodType(self.disable_quantization, model)

        model.enable_quantization()

        return model

    def convert_layers(self, module):
        # for name, param in module.named_parameters():
        #     print(f"Parameter: {name}")
        #     print(f"Shape: {param.shape}")

        # assert 0==1

        for name, child in module.named_children():
            if isinstance(child, (QTLinear, QTConv2d)):
                print('Error! Already have QModules!!!')
                continue

            if isinstance(child, nn.Linear):
                # print("child.weight.shape[1]", child.weight.shape[1])  # 1000
                # print("child.weight.shape[0]", child.weight.shape[0])  # 512
                new_child = QTLinear(child.weight.shape[1], child.weight.shape[0], #### 代替方法！！！
                                     child.bias is not None)
                new_child.weight = child.weight

                if child.bias is not None:
                    new_child.bias = child.bias
                setattr(module, name, new_child)
            elif isinstance(child, nn.Conv2D):
                new_child = QTConv2d(child._in_channels, child._out_channels,
                                     child._kernel_size, child._stride, child._padding, 
                                     child._dilation, child._groups, child.bias is not None)
                new_child.weight = child.weight
                if child.bias is not None:
                    new_child.bias = child.bias
                setattr(module, name, new_child)
            else:
                self.convert_layers(child)

    def parse_quantization_params(self, model):
        if self.qconfig.has_key('weight_config'):
            if self.qconfig.has_key('weight_forward_config') or self.qconfig.has_key('weight_backward_config'):
                raise KeyError(f'can not use weight_config and weight_forward_config \
                                 or weight_backward_config at the same time')
            weight_forward_config = weight_backward_config = self.qconfig.weight_config
        else:
            assert self.qconfig.has_key('weight_forward_config') and self.qconfig.has_key('weight_backward_config')
            weight_forward_config = self.qconfig.weight_forward_config
            weight_backward_config = self.qconfig.weight_backward_config
        
        if self.qconfig.has_key('input_config'):
            if self.qconfig.has_key('input_forward_config') or self.qconfig.has_key('input_backward_config'):
                raise KeyError(f'can not use input_config and input_forward_config \
                                 or input_backward_config at the same time')
            input_forward_config = input_backward_config = self.qconfig.input_config
        else:
            assert self.qconfig.has_key('input_forward_config') and self.qconfig.has_key('input_backward_config')
            input_forward_config = self.qconfig.input_forward_config
            input_backward_config = self.qconfig.input_backward_config

        if self.qconfig.has_key('grad_config'):
            if self.qconfig.has_key('grad_input_config') or self.qconfig.has_key('grad_weight_config'):
                raise KeyError(f'can not use grad_config and grad_input_config \
                                 or grad_weight_config at the same time')
            grad_input_config = grad_weight_config = self.qconfig.grad_config
        else:
            assert self.qconfig.has_key('grad_input_config') and self.qconfig.has_key('grad_weight_config')
            grad_input_config = self.qconfig.grad_input_config
            grad_weight_config = self.qconfig.grad_weight_config

        # collect default quantization information for each module
        module_list = []
        for m_name, m in model.named_sublayers():
            if isinstance(m, (nn.Linear, nn.Conv2D)):
                qparam = Dict()
                qparam.name = m_name
                for var in ['weight_forward', 'weight_backward', 'input_forward', 'input_backward', 'grad_input', 'grad_weight']:
                    var_config = eval(var + '_config')
                    if var_config.qtype == 'full':
                        qparam[var] = Dict({
                            'qtype': var_config.qtype,
                            'granularity': None,
                            'scaling': None,
                            'observer': None,
                            'stochastic': None})
                    else:
                        qparam[var] = Dict({
                            'qtype': var_config.qtype,
                            'granularity': var_config.granularity,
                            'scaling': var_config.scaling,
                            'observer': var_config.observer,
                            'stochastic': var_config.stochastic})
                module_list.append(qparam)

        # Figure out non-default quantization config for specific modules according to module index/name
        for var in ['weight_forward', 'weight_backward', 'input_forward', 'input_backward', 'grad_input', 'grad_weight']:
            var_config = eval(var + '_config')
            if var_config.has_key('filters'):
                for filter_item in var_config['filters']:
                    if not isinstance(filter_item, dict):
                        raise TypeError(f'filter item must be a dict, but got {type(filter_item)}')

                    if filter_item.has_key('id'):
                        for key in ['qtype', 'granularity', 'scaling', 'observer', 'stochastic']:
                            if filter_item.has_key(key):
                                module_list[filter_item['id']][var][key] = filter_item[key]
                    elif filter_item.has_key('name'):
                        for key in ['qtype', 'granularity', 'scaling', 'observer', 'stochastic']:
                            if filter_item.has_key(key):
                                for m in module_list:
                                    if filter_item['name'] in m['name']:
                                        m[var][key] = filter_item[key]
                    else:
                        raise KeyError('only support filter with id or module name')

        idx = 0
        for m_name, m in model.named_sublayers():
            if isinstance(m, (nn.Linear, nn.Conv2D)):
                m.init(module_list[idx])
                idx += 1
