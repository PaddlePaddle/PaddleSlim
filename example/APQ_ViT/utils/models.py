import sys
import paddle
import paddle.nn as nn
from types import MethodType
import paddle.nn.functional as F


def paddle_forward(self, x):
    qkv = self.qkv(x).chunk(3, axis=-1)
    q, k, v = map(self.transpose_multihead, qkv)

    q = q * self.scales
    # import pdb; pdb.set_trace()
    attn = self.matmul1(q, k.transpose([0, 1, 3, 2]))  # [B, n_heads, N, N]
    attn = self.softmax(attn)
    attn = self.attn_dropout(attn)

    z = self.matmul2(attn, v)  # [B, n_heads, N, head_dim]
    z = z.transpose([0, 2, 1, 3])  # [B, N, n_heads, head_dim]
    new_shape = z.shape[:-2] + [self.all_head_size]
    z = z.reshape(new_shape)  # [B, N, all_head_size]

    z = self.out(z)
    z = self.proj_dropout(z)
    return z


def attention_forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
    q, k, v = paddle.unbind(qkv, axis=0)  # equivalent to unbind in PyTorch

    # attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = self.matmul1(q, k.transpose([0, 1, 3, 2])) * self.scales
    attn = F.softmax(attn, axis=-1)
    attn = self.attn_dropout(attn)
    
    # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.matmul2(attn, v).transpose([0, 2, 1, 3]).reshape([B, N, C])
    x = self.out(x)
    x = self.proj_dropout(x)
    return x


def window_attention_forward(self, x, mask=None):
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape([B_, N, 3, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
    q, k, v = paddle.unbind(qkv, axis=0) # equivalent to unbind in PyTorch

    q = q * self.scales
    # attn = (q @ k.transpose(-2, -1))
    attn = self.matmul1(q, k.transpose([0, 1, 3, 2]))

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape([-1])].reshape(
    [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])  # Wh*Ww, Wh*Ww, nH
    relative_position_bias = relative_position_bias.transpose((2, 0, 1)).contiguous()  # nH, Wh*Ww, Wh*Ww
 
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.reshape([B_ // nW, nW, self.num_heads, N, N]) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.reshape([-1, self.num_heads, N, N])
        attn = attn.softmax(axis=-1)
    else:
        attn = attn.softmax(axis=-1)

    attn = self.attn_drop(attn)

    # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.matmul2(attn, v).transpose([0, 2, 1, 3]).reshape([B_, N, C])
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

class MatMul(nn.Layer):
    def forward(self, A, B):
        return paddle.matmul(A, B)


def get_net(name, config_path=None, model_path=None):
    """
    Get a vision transformer model in Paddle.
    This function replaces matrix multiplication operations with matmul modules in the model.
    """

    # Create the model
    # net = timm.create_model(name, pretrained=True)
    from .config import get_config
    from .vit import build_vit as build_model
    from .vit import Attention

    # config files in ./configs/
    # config = get_config('/mnt/disk1/jsh/2030/paddle/apq-vit/paddle_project/utils/configs/vit_small_patch32_224.yaml')
    config = get_config(config_path)
    # build model
    model = build_model(config)
    # load pretrained weights
    # model_state_dict = paddle.load('/mnt/disk1/jsh/2030/paddle/apq-vit/paddle_project/utils/weights/vit_small_patch32_224.pdparams')
    model_state_dict = paddle.load(model_path)
    
    model.set_state_dict(model_state_dict)

    for name, module in model.named_sublayers():
        if isinstance(module, Attention):
            # import pdb; pdb.set_trace()
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(paddle_forward, module)
        # if isinstance(module, WindowAttention):
        #     setattr(module, "matmul1", MatMul())
        #     setattr(module, "matmul2", MatMul())
        #     module.forward = window_attention_forward.__get__(module)
    
    model.eval()  # Set the model to evaluation mode
    return config, model
