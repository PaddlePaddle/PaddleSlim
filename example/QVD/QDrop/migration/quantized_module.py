import sys
import paddle
import copy
import gc
from .observer import AvgTokenQuantileObserver, MinMaxObserver, EMAMinMaxObserver, AvgMinMaxObserver, MSEObserver, AvgMSEObserver, MSEFastObserver, AvgMSEFastObserver, EMAMSEFastObserver, EMAQuantileObserver, AvgQuantileObserver, LSQPlusObserver
from .fake_quant import FixedFakeQuantize, GroupFixedFakeQuantize, TokenFixedFakeQuantize, FixedQuantize, GroupFixedQuantize
ObserverDict = {'MinMaxObserver': MinMaxObserver, 'EMAMinMaxObserver':
    EMAMinMaxObserver, 'AvgMinMaxObserver': AvgMinMaxObserver,
    'MSEObserver': MSEObserver, 'AvgMSEObserver': AvgMSEObserver,
    'MSEFastObserver': MSEFastObserver, 'AvgMSEFastObserver':
    AvgMSEFastObserver, 'EMAMSEFastObserver': EMAMSEFastObserver,
    'AvgQuantileObserver': AvgQuantileObserver, 'EMAQuantileObserver':
    EMAQuantileObserver, 'LSQPlusObserver': LSQPlusObserver,
    'AvgTokenQuantileObserver': AvgTokenQuantileObserver}
FakeQuantizeDict = {'FixedFakeQuantize': FixedFakeQuantize,
    'GroupFixedFakeQuantize': GroupFixedFakeQuantize,
    'TokenFixedFakeQuantize': TokenFixedFakeQuantize, 'FixedQuantize':
    FixedQuantize, 'GroupFixedQuantize': GroupFixedQuantize}


class QuantizedModule(paddle.nn.Layer):

    def __init__(self, backend='academic'):
        super().__init__()
        self.cac_migrate = False
        self.backend = backend

    def set_cac_migrate(self, state):
        self.cac_migrate = state


class QuantizedOperator:
    pass


class QConv2d(QuantizedOperator, paddle.nn.Conv2D):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, dilation, groups, bias, padding_mode, w_qconfig):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias, padding_mode=
            padding_mode)
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight
            ), self.bias)


class QLinear(QuantizedOperator, paddle.nn.Linear):

    def __init__(self, in_features, out_features, bias, w_qconfig):
        super().__init__(in_features=in_features, out_features=out_features,
            bias=bias)
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input):
        return paddle.nn.functional.linear(x=input, weight=self.
            weight_fake_quant(self.weight).T, bias=self.bias)


import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class WeightQuantizer(nn.Layer):
    def __init__(self, w_qconfig):
        super().__init__()
        self.w_qconfig = w_qconfig
        # Define any additional logic for WeightQuantizer here if necessary

    def forward(self, weight):
        # Implement weight quantization logic
        # For simplicity, assuming it returns the weight as is in this case
        return weight


class QEmbedding(nn.Embedding):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx,
                 max_norm,
                 norm_type,
                 scale_grad_by_freq,
                 sparse,
                 _weight,
                 w_qconfig):
        super().__init__(num_embeddings=num_embeddings,
                         embedding_dim=embedding_dim,
                         padding_idx=padding_idx,
                         max_norm=max_norm,
                         norm_type=norm_type,
                         scale_grad_by_freq=scale_grad_by_freq,
                         sparse=sparse,
                         weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(_weight)))
        
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input):
        return F.embedding(
            input, self.weight_fake_quant(self.weight), self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


module_type_to_quant_weight = {paddle.nn.Linear: QLinear, paddle.nn.Conv2D:
    QConv2d, paddle.nn.Embedding: QEmbedding}


def get_module_args(module):
    if isinstance(module, paddle.nn.Linear):
        return dict(in_features=module.in_features, out_features=module.
            out_features, bias=module.bias is not None)
    elif isinstance(module, paddle.nn.Conv2D):
        return dict(in_channels=module.in_channels, out_channels=module.
            out_channels, kernel_size=module.kernel_size, stride=module.
            stride, padding=module.padding, dilation=module.dilation,
            groups=module.groups, bias=module.bias is not None,
            padding_mode=module.padding_mode)
    elif isinstance(module, paddle.nn.Embedding):
        return dict(num_embeddings=module.num_embeddings, embedding_dim=
            module.embedding_dim, padding_idx=module.padding_idx, max_norm=
            module.max_norm, norm_type=module.norm_type, scale_grad_by_freq
            =module.scale_grad_by_freq, sparse=module.sparse, _weight=None)
    else:
        raise NotImplementedError


def Quantizer(module, config):
    if module is None:
        return ActivationQuantizer(a_qconfig=config)
    module_type = type(module)
    if module_type in module_type_to_quant_weight:
        kwargs = get_module_args(module)
        qmodule = module_type_to_quant_weight[module_type](**kwargs,
            w_qconfig=config)
        qmodule.weight = module.weight
        if getattr(module, 'bias', None) is not None:
            qmodule.bias = module.bias
        del module
        gc.collect()
        return qmodule
    return module


class QuantizedLayer(QuantizedModule):

    def __init__(self, module, activation, w_qconfig, a_qconfig, qinput=True):
        super().__init__()
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
        if qinput:
            self.act_fake_quant = Quantizer(None, a_qconfig)
        self.qinput = qinput
        self.module = Quantizer(module, w_qconfig)
        self.activation = activation

    def forward(self, x, observation_mask=None, seq_pos=-1):
        if self.qinput:
            x = self.act_fake_quant(x, observation_mask, seq_pos)
        x = self.module(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def ActivationQuantizer(a_qconfig):
    if 'Group' in a_qconfig.quantizer:
        return FakeQuantizeDict[a_qconfig.quantizer](ObserverDict[a_qconfig
            .observer], bit=a_qconfig.bit, symmetric=a_qconfig.symmetric,
            ch_axis=a_qconfig.ch_axis, group_size=a_qconfig.group_size)
    return FakeQuantizeDict[a_qconfig.quantizer](ObserverDict[a_qconfig.
        observer], bit=a_qconfig.bit, symmetric=a_qconfig.symmetric,
        ch_axis=a_qconfig.ch_axis)


def WeightQuantizer(w_qconfig):
    if 'Group' in w_qconfig.quantizer:
        return FakeQuantizeDict[w_qconfig.quantizer](ObserverDict[w_qconfig
            .observer], bit=w_qconfig.bit, symmetric=w_qconfig.symmetric,
            ch_axis=w_qconfig.ch_axis, group_size=w_qconfig.group_size)
    return FakeQuantizeDict[w_qconfig.quantizer](ObserverDict[w_qconfig.
        observer], bit=w_qconfig.bit, symmetric=w_qconfig.symmetric,
        ch_axis=w_qconfig.ch_axis)
