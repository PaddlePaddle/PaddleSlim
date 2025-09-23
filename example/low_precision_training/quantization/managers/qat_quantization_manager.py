import paddle
import paddle.nn as nn
import abc
import types
import copy

from .base_quantization_manager import BaseQuantizationManager
from . import register_quantization_manager
from ..layers import QLinear, QConv2d

@register_quantization_manager('qat_quantization_manager')
class QATQuantizationManager(BaseQuantizationManager):
    def __init__(self, qconfig):
        super(QATQuantizationManager, self).__init__(qconfig)
        self.qconfig = qconfig

    def prepare(self, model, inplace=True):
        if not inplace:
            model = copy.deepcopy(model)

        # convert Conv2d -> QConv2d, etc
        self.convert_layers(model)

        # set bitwidth information for each module
        self.parse_quantization_params(model)

        # attach calibrating func
        def calibrating(model, state):
            for m in model.sublayers():
                if isinstance(m, (QLinear, QConv2d)):
                    m.calibrating = state
        model.calibrating = types.MethodType(calibrating, model)

        return model

    def initialize_quantizer(self, model):
        for m in model.sublayers():
            if isinstance(m, (QLinear, QConv2d)):
                m.initialize_quantizer()

    def convert_layers(self, module):
        for name, child in module.named_sublayers():
            if isinstance(child, (QLinear, QConv2d)):
                print('Error! Already have QModules!!!')
                continue

            if isinstance(child, nn.Linear):
                new_child = QLinear(child.weight.shape[0], child.weight.shape[1],
                                     child.bias is not None)
                new_child.weight = child.weight
                if child.bias is not None:
                    new_child.bias = child.bias
                setattr(module, name, new_child)
            elif isinstance(child, nn.Conv2D):
                new_child = QConv2d(child.weight.shape[0], child.weight.shape[1],
                                     child._kernel_size, child._stride, child._padding, 
                                     child._dilation, child._groups, child.bias is not None)
                new_child.weight = child.weight
                if child.bias is not None:
                    new_child.bias = child.bias
                setattr(module, name, new_child)
            else:
                self.convert_layers(child)

    def parse_quantization_params(self, model):
        # collect default quantization information for each module
        module_list = []
        for m_name, m in model.named_sublayers():
            if isinstance(m, (nn.Linear, nn.Conv2D)):
                # config for input activations
                qparam = {
                    'name': m_name,
                    'input_qtype': self.qconfig.activation_config.qtype,
                    'input_granularity': self.qconfig.activation_config.granularity,
                    'weight_qtype': self.qconfig.weight_config.qtype,
                    'weight_granularity': self.qconfig.weight_config.granularity
                }
                module_list.append(qparam)

        # Figure out non-default quantization config for specific modules according to module index/name
        # For activations
        if self.qconfig.activation_config.has_key("filters"):
            for filter_item in self.qconfig.activation_config.filters:
                if not isinstance(filter_item, dict):
                    raise TypeError(f'filter item must be a dict, but got {type(filter_item)}')

                if filter_item.has_key('id'):
                    if filter_item.has_key('qtype'):
                        module_list[filter_item['id']]['input_qtype'] = filter_item['qtype']
                    if filter_item.has_key('granularity'):
                        module_list[filter_item['id']]['input_granularity'] = filter_item['granularity']
                elif filter_item.has_key('name'):
                    if filter_item.has_key('qtype'):
                        for m in module_list:
                            if filter_item['name'] in m['name']:
                                m['input_qtype'] = filter_item['qtype']
                    if filter_item.has_key('granularity'):
                        for m in module_list:
                            if filter_item['name'] in m['name']:
                                m['input_granularity'] = filter_item['granularity']
                else:
                    raise KeyError(f'only support filter with id or module name, but got {filter_item.keys()}')
        
        # For weights
        if self.qconfig.weight_config.has_key("filters"):
            for filter_item in self.qconfig.weight_config.filters:
                if not isinstance(filter_item, dict):
                    raise TypeError(f'filter item must be a dict, but got {type(filter_item)}')

                if filter_item.has_key('id'):
                    if filter_item.has_key('qtype'):
                        module_list[filter_item['id']]['weight_qtype'] = filter_item['qtype']
                    if filter_item.has_key('granularity'):
                        module_list[filter_item['id']]['weight_granularity'] = filter_item['granularity']
                elif filter_item.has_key('name'):
                    if filter_item.has_key('qtype'):
                        for m in module_list:
                            if filter_item['name'] in m['name']:
                                m['weight_qtype'] = filter_item['qtype']
                    if filter_item.has_key('granularity'):
                        for m in module_list:
                            if filter_item['name'] in m['name']:
                                m['weight_granularity'] = filter_item['granularity']
                else:
                    raise KeyError(f'only support filter with id or module name, but got {filter_item.keys()}')

        idx = 0
        for m_name, m in model.named_sublayers():
            if isinstance(m, (nn.Linear, nn.Conv2D)):
                m.init(module_list[idx])
                idx += 1