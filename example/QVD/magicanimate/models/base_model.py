import paddle
import paddle.nn as nn
import paddle.tensor as tensor
from easydict import EasyDict

from QDrop.migration.migration import migration
from QDrop.migration.util_layernorm import Identity
from QDrop.quant.quant_layer import QuantModule
from QDrop.quant.quant_block import QuantLoRACompatibleLinear

from ppdiffusers.models.attention_processor import Attention as CrossAttention
from ppdiffusers.models.lora import LoRACompatibleLinear


from magicanimate.models.orig_attention import CrossAttention as VersatileAttention
from magicanimate.models.diffusers_attention import FeedForward
import logging

logger = logging.getLogger('QVD')
MODE = 'mid'
do_scale = False

a_dict = {
    'quantizer': 'FixedFakeQuantize',
    'observer': 'AvgMSEObserver',
    'symmetric': True,
    'ch_axis': -1,
    'bit': 8
}
w_dict = {
    'quantizer': 'FixedFakeQuantize',
    'observer': 'MinMaxObserver',
    'symmetric': False,
    'ch_axis': 0,
    'bit': 8
}
a_qconfig = EasyDict(a_dict)
w_qconfig = EasyDict(w_dict)

class BaseModel(nn.Layer):
    def __init__(self):
        super().__init__()
        self.cac_migrate = True
        self.identity = {}

    def set_cac_migrate(self, state):
        self.cac_migrate = state

    def migration_conv(self, hidden_states, layer: nn.Conv2D, name):
        w_qconfig.ch_axis = 1
        if MODE == 'mid':
            ori_shape = hidden_states.shape
            cmn = tensor.min(hidden_states.transpose([1, 0, 2, 3]).reshape([ori_shape[1], -1]), axis=1).reshape([1, ori_shape[1], 1, 1])
            cmx = tensor.max(hidden_states.transpose([1, 0, 2, 3]).reshape([ori_shape[1], -1]), axis=1).reshape([1, ori_shape[1], 1, 1])
            shift = (cmn + cmx) / 2
        elif MODE == 'mean':
            shift = paddle.mean(hidden_states, axis=(-2, -1), keepdim=True).mean(axis=0, keepdim=True)
        else:
            raise NotImplementedError

        self.identity[name] = Identity()
        self.identity[name].set_migrate_bias(shift)
        hidden_states -= shift

        layer.bias.set_value(layer(shift.tile([1, 1, layer.weight.shape[2], layer.weight.shape[3]])).squeeze())

        extra_dict = {
            'bias': layer.bias,
            'shift': shift,
            'conv_fwd_kwargs': getattr(layer, 'fwd_kwargs', {})
        }

        if do_scale:
            best_scale = migration(hidden_states, layer, a_qconfig, w_qconfig, 'conv', extra_dict)
            self.identity[name].set_migrate_scale(best_scale)
            layer.weight.set_value(layer.weight * best_scale)
            hidden_states /= best_scale

        return hidden_states

    def migration_FFN(self, hidden_states, ffn: FeedForward, name):
        w_qconfig.ch_axis = 0
        if MODE == 'mid':
            cmx = tensor.max(hidden_states, axis=(0, 1))
            cmn = tensor.min(hidden_states, axis=(0, 1))
            shift = (cmx + cmn) / 2
        elif MODE == 'mean':
            shift = paddle.mean(hidden_states, axis=(0, 1))
        
        self.identity[name] = Identity()
        self.identity[name].set_migrate_bias(shift)

        hidden_states -= shift

        if ffn.net[0].proj.linear.bias is None:
            ffn.net[0].proj.linear.bias = paddle.create_parameter(
                shape=[ffn.out_features], dtype=ffn.net[0].proj.linear.weight.dtype, default_initializer=paddle.nn.initializer.Constant(0))

        ffn.net[0].proj.linear.bias.set_value(ffn.net[0].proj.linear.bias + shift @ ffn.net[0].proj.linear.weight.transpose([1, 0]))

        extra_dict = {'bias': None, 'shift': shift}
        if do_scale:
            best_scale = migration(hidden_states, ffn, a_qconfig, w_qconfig, 'fc1', extra_dict=extra_dict)
            self.identity[name].set_migrate_scale(best_scale)
            hidden_states /= best_scale
            ffn.net[0].proj.linear.weight.set_value(ffn.net[0].proj.linear.weight * best_scale)

        return hidden_states

    def migration_Linear(self, hidden_states, linear, name):
        w_qconfig.ch_axis = 0
        if MODE == 'mid':
            cmx = tensor.max(hidden_states, axis=(0, 1))
            cmn = tensor.min(hidden_states, axis=(0, 1))
            shift = (cmx + cmn) / 2
        elif MODE == 'mean':
            shift = paddle.mean(hidden_states, axis=(0, 1))

        self.identity[name] = Identity()
        self.identity[name].set_migrate_bias(shift)
        hidden_states -= shift

        if isinstance(linear, QuantModule):
            if linear.bias is None:
                linear.bias = paddle.create_parameter(
                    shape=[linear.weight.shape[0]], dtype=linear.weight.dtype, default_initializer=paddle.nn.initializer.Constant(0))
            linear.bias.set_value(linear.bias + shift @ linear.weight.transpose([1, 0]))
        elif isinstance(linear, (QuantLoRACompatibleLinear, LoRACompatibleLinear)):
            if linear.linear.bias is None:
                linear.linear.bias = paddle.create_parameter(
                    shape=[linear.linear.weight.shape[0]], dtype=linear.linear.weight.dtype, default_initializer=paddle.nn.initializer.Constant(0))
            linear.linear.bias.set_value(linear.linear.bias + shift @ linear.linear.weight.transpose([1, 0]))
        else:
            raise NotImplementedError

        extra_dict = {'bias': None, 'shift': shift}
        if do_scale:
            best_scale = migration(hidden_states, linear, a_qconfig, w_qconfig, 'fc1', extra_dict=extra_dict)
            self.identity[name].set_migrate_scale(best_scale)
            hidden_states /= best_scale
            if isinstance(linear, QuantModule):
                linear.weight.set_value(linear.weight * best_scale)
            elif isinstance(linear, QuantLoRACompatibleLinear):
                linear.linear.weight.set_value(linear.linear.weight * best_scale)

        return hidden_states

    def migration_CrossAttention(self, hidden_states, attn: CrossAttention, name):
        w_qconfig.ch_axis = 0
        if MODE == 'mid':
            cmx = tensor.max(hidden_states, axis=(0, 1))
            cmn = tensor.min(hidden_states, axis=(0, 1))
            shift = (cmx + cmn) / 2
        elif MODE == 'mean':
            shift = paddle.mean(hidden_states, axis=(0, 1))

        self.identity[name] = Identity()
        self.identity[name].set_migrate_bias(shift)
        hidden_states -= shift

        if attn.to_k.linear.bias is None:
            dtype = attn.to_k.linear.weight.dtype
            device = attn.to_k.linear.weight.place
            attn.to_k.linear.bias = paddle.create_parameter(shape=[attn.inner_dim], dtype=dtype, default_initializer=paddle.nn.initializer.Constant(0))
            attn.to_q.linear.bias = paddle.create_parameter(shape=[attn.inner_dim], dtype=dtype, default_initializer=paddle.nn.initializer.Constant(0))
            attn.to_v.linear.bias = paddle.create_parameter(shape=[attn.inner_dim], dtype=dtype, default_initializer=paddle.nn.initializer.Constant(0))

        attn.to_q.linear.bias.set_value(attn.to_q.linear.bias + shift @ attn.to_q.linear.weight.transpose([1, 0]))
        attn.to_k.linear.bias.set_value(attn.to_k.linear.bias + shift @ attn.to_k.linear.weight.transpose([1, 0]))
        attn.to_v.linear.bias.set_value(attn.to_v.linear.bias + shift @ attn.to_v.linear.weight.transpose([1, 0]))

        extra_dict = {
            'bias': None,
            'num_heads': attn.heads,
            'head_dim': int(attn.inner_dim / attn.heads),
            'scaling': attn.scale,
            'attention_mask': None,
            'shift': shift
        }

        if do_scale:
            best_scale = migration(hidden_states, attn, a_qconfig, w_qconfig, 'qkv', extra_dict=extra_dict)
            self.identity[name].set_migrate_scale(best_scale)
            hidden_states /= best_scale
            attn.to_k.linear.weight.set_value(attn.to_k.linear.weight * best_scale)
            attn.to_q.linear.weight.set_value(attn.to_q.linear.weight * best_scale)
            attn.to_v.linear.weight.set_value(attn.to_v.linear.weight * best_scale)

        return hidden_states

    def migration_LinearAttention(self, hidden_states, attn: VersatileAttention, name, encoder_hidden_states=None):
        w_qconfig.ch_axis = 0

        if MODE == 'mid':
            cmx = tensor.max(hidden_states, axis=(0, 1))
            cmn = tensor.min(hidden_states, axis=(0, 1))
            shift = (cmx + cmn) / 2
        elif MODE == 'mean':
            shift = paddle.mean(hidden_states, axis=(0, 1))
        else:
            raise NotImplementedError

        self.identity[name] = Identity()
        self.identity[name].set_migrate_bias(shift)
        hidden_states -= shift

        if attn.to_k.bias is None:
            attn.to_k.bias = paddle.create_parameter(shape=[attn.inner_dim], dtype=attn.to_k.weight.dtype, default_initializer=paddle.nn.initializer.Constant(0))
            attn.to_q.bias = paddle.create_parameter(shape=[attn.inner_dim], dtype=attn.to_q.weight.dtype, default_initializer=paddle.nn.initializer.Constant(0))
            attn.to_v.bias = paddle.create_parameter(shape=[attn.inner_dim], dtype=attn.to_v.weight.dtype, default_initializer=paddle.nn.initializer.Constant(0))

        attn.to_q.bias.set_value(attn.to_q.bias + shift @ attn.to_q.weight.transpose([1, 0]))
        attn.to_k.bias.set_value(attn.to_k.bias + shift @ attn.to_k.weight.transpose([1, 0]))
        attn.to_v.bias.set_value(attn.to_v.bias + shift @ attn.to_v.weight.transpose([1, 0]))

        extra_dict = {
            'bias': None,
            'num_heads': attn.heads,
            'head_dim': int(attn.inner_dim / attn.heads),
            'scaling': attn.scale,
            'attention_mask': None,
            'shift': shift
        }

        if do_scale:
            best_scale = migration(hidden_states, attn, a_qconfig, w_qconfig, 'qkv', extra_dict=extra_dict)
            self.identity[name].set_migrate_scale(best_scale)
            hidden_states /= best_scale
            attn.to_k.weight.set_value(attn.to_k.weight * best_scale)
            attn.to_q.weight.set_value(attn.to_q.weight * best_scale)
            attn.to_v.weight.set_value(attn.to_v.weight * best_scale)

        return hidden_states
