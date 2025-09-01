import paddle
from paddle import nn
from quant_layers.linear import *
from quant_layers.matmul import *
from quant_layers.conv import *
from models.vit import Attention
from models.swin import WindowAttention
from types import MethodType


class MatMul(nn.Layer):
    def forward(self, x,y,
              transpose_x=False,
              transpose_y=False,
            ):
        return paddle.matmul(self, x,y,transpose_x,transpose_y)


def vit_attn_forward(self, x):
    qkv = self.qkv(x).chunk(3, axis=-1)
    q, k, v = map(self.transpose_multihead, qkv)

    q = q * self.scales
    k=k.transpose([0,1,3,2]) #为了统一量化参数channelwise的处理，提前transpose,否则paddle是放在matmul里transpose
    attn = self.matmul1(q, k)  # [B, n_heads, N, N]
    attn = self.softmax(attn)
    attn = self.attn_dropout(attn)

    z = self.matmul2(attn, v)  # [B, n_heads, N, head_dim]
    z = z.transpose([0, 2, 1, 3])  # [B, N, n_heads, head_dim]
    new_shape = z.shape[:-2] + [self.all_head_size]
    z = z.reshape(new_shape)  # [B, N, all_head_size]

    z = self.out(z)
    z = self.proj_dropout(z)
    return z


def swin_attn_forward(self, x, mask=None):
    qkv = self.qkv(x).chunk(3, axis=-1)  # list of 3 elements
    q, k, v = map(self.transpose_multihead, qkv)
    q = q * self.scale
    k=k.transpose([0,1,3,2]) #为了统一量化参数channelwise的处理，提前transpose,否则paddle是放在matmul里transpose
    attn = self.matmul1(q, k)

    relative_position_bias = self.get_relative_pos_bias_from_pos_index()

    relative_position_bias = relative_position_bias.reshape(
        [self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1])

    # nH, window_h*window_w, window_h*window_w
    relative_position_bias = relative_position_bias.transpose([2, 0, 1])
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.reshape(
            [x.shape[0] // nW, nW, self.num_heads, x.shape[1], x.shape[1]])
        attn += mask.unsqueeze(1).unsqueeze(0)
        attn = attn.reshape([-1, self.num_heads, x.shape[1], x.shape[1]])
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)

    attn = self.attn_dropout(attn)

    z = self.matmul2(attn, v)
    z = z.transpose([0, 2, 1, 3])
    new_shape = z.shape[:-2] + [self.dim]
    z = z.reshape(new_shape)
    z = self.proj(z)
    z = self.proj_dropout(z)

    return z
    
    
def wrap_modules_in_net(model, cfg, reparam=False, recon=False):
    for name, module in model.named_sublayers(include_self=True):
        if isinstance(module, Attention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(vit_attn_forward, module)
        if isinstance(module, WindowAttention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(swin_attn_forward, module)

    module_dict={}
    for name, module in model.named_sublayers(include_self=True):
        module_dict[name] = module
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")

        if isinstance(module, nn.Conv2D):
            idx = idx + 1 if idx != 0 else idx
            new_module = AsymmetricallyBatchingQuantConv2d(
                in_channels = module._in_channels, 
                out_channels = module._out_channels,
                kernel_size = module._kernel_size,
                stride = module._stride,
                qmode = 'raw',
                w_bit = cfg.w_bit,
                a_bit = cfg.qconv_a_bit,
                metric = cfg.calib_metric,
                calib_batch_size = cfg.calib_batch_size,
                search_round = cfg.search_round,
                eq_n = cfg.eq_n,
            )
            new_module.weight.data = paddle.assign(module.weight.data)
            new_module.bias.data = paddle.assign(module.bias.data)
            setattr(father_module, name[idx:], new_module)
        if isinstance(module, MatMul):
            idx = idx + 1 if idx != 0 else idx
            new_module = AsymmetricallyBatchingQuantMatMul(
                A_bit = cfg.a_bit,
                B_bit = cfg.a_bit,
                qmode = 'raw',
                metric = cfg.calib_metric,
                calib_batch_size = cfg.calib_batch_size,
                search_round = cfg.search_round,
                eq_n = cfg.eq_n,
                head_channel_wise = cfg.matmul_head_channel_wise,
                token_channel_wise = cfg.token_channel_wise,
                num_heads = father_module.num_heads,
            )
            setattr(father_module, name[idx:], new_module)
        if isinstance(module, nn.Linear):
            cur_a_bit = cfg.qhead_a_bit if ('classifier' in name or 'pre_logits' in name ) else cfg.a_bit
            linear_kwargs = {
                'in_features': module.weight.shape[0],
                'out_features': module.weight.shape[1],
                'bias': module.bias is not None,
                'qmode': 'raw',
                'w_bit': cfg.w_bit,
                'a_bit': cur_a_bit,
                'metric': cfg.calib_metric,
                'calib_batch_size': cfg.calib_batch_size,
                'search_round': cfg.search_round,
                'eq_n': cfg.eq_n,
                'n_V': 3 if 'qkv' in name else 1,
                'token_channel_wise': cfg.token_channel_wise,
                'weight_attr':module._weight_attr,
                'bias_attr':module._bias_attr,
            }
            idx = idx + 1 if idx != 0 else idx
            if cur_a_bit == cfg.w_bit and reparam and ('qkv' in name or 'reduction' in name or 'fc1' in name):
                idxx = father_name.rfind('.')
                idxx = 0 if idxx == -1 else idxx
                grandfather_name = father_name[:idxx]
                if grandfather_name in module_dict:
                    grandfather_module = module_dict[grandfather_name]
                new_module = AsymmetricallyChannelWiseBatchingQuantLinear(
                    **linear_kwargs, 
                )
                if 'qkv' in name:
                    new_module.prev_layer = grandfather_module.attn_norm
                if 'fc1' in name:
                    new_module.prev_layer = grandfather_module.mlp_norm
                if 'reduction' in name:
                    new_module.prev_layer = father_module.norm
            else: 
                new_module = AsymmetricallyBatchingQuantLinear(
                    **linear_kwargs,
                    post_relu = ('fc2' in name and recon),
                )
            new_module.weight.data = paddle.assign(module.weight.data)
            if module.bias is not None:
                new_module.bias.data = paddle.assign(module.bias.data)
            setattr(father_module, name[idx:], new_module)
    return model


def wrap_reparamed_modules_in_net(model):
    module_dict = {}
    for name, module in model.named_sublayers(include_self=True):
        module_dict[name] = module
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")

        if isinstance(module, AsymmetricallyChannelWiseBatchingQuantLinear):
            idx = idx + 1 if idx != 0 else idx
            linear_kwargs = {
                'in_features': module.weight.shape[0],
                'out_features': module.weight.shape[1],
                'bias': module.bias is not None,
                'qmode': module.qmode,
                'w_bit': module.w_quantizer.n_bits,
                'a_bit': module.a_quantizer.n_bits,
                'metric': module.metric,
                'calib_batch_size': module.calib_batch_size,
                'search_round': module.search_round,
                'eq_n': module.eq_n,
                'n_V': module.n_V,
                'token_channel_wise': module.token_channel_wise,
            }
            new_module = AsymmetricallyBatchingQuantLinear(**linear_kwargs)
            if (new_module.a_quantizer.scale.shape != module.a_quantizer.scale.shape):
                new_module.a_quantizer.scale.data = module.a_quantizer.scale.data.clone()
            state_dict =  module.state_dict()
            for key in state_dict:
                #print(key,state_dict[key].data.dtype)
                state_dict[key] = state_dict[key].astype(paddle.float32)
            new_module.set_state_dict(module.state_dict())
            new_module.calibrated = True
            new_module.a_quantizer.inited = True
            new_module.w_quantizer.inited = True
            setattr(father_module, name[idx:], new_module)
    return model
    