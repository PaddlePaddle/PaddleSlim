import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Union
import time
import warnings
from .Qlog import LogQuantizer
from .UniformAffineQuantizer import UniformAffineQuantizer, StraightThrough

def lp_loss(pred, tgt, p=2.0, reduction="none"):
    """
    loss function measured in L_p Norm
    """
    if reduction == "none":
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()

def round_ste(x: paddle.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


class QuantModule(nn.Layer):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: Union[nn.Conv2D, nn.Linear],
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_act_quant=False,
        in_channels=1,
        act_quantizer='uniform'
    ):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2D):
            self.fwd_kwargs = dict(
                stride=org_module._stride,
                padding=org_module._padding,
                dilation=org_module._dilation,
                groups=org_module._groups,
            )
            self.fwd_func = F.conv2d
            self.name = "conv2d"
            in_channels = org_module._in_channels
            shape = 4
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
            self.name = "linear"
            in_channels = org_module.weight.shape[0]
            shape = 2

        self.weight = org_module.weight
        if org_module.bias is not None:
            self.bias = org_module.bias
        else:
            self.bias = None

        self.use_weight_quant = False
        self.use_act_quant = False
        weight_quant_params["in_channels"] = in_channels
        weight_quant_params["shape"] = shape

        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = self.get_quantizer(act_quantizer, act_quant_params)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        self.disable_act_quant = disable_act_quant

    def get_quantizer(self, name, quant_params):
        if name == 'uniform':
            return UniformAffineQuantizer(**quant_params)
        elif name == 'log':
            return LogQuantizer(**quant_params)
        else:
            raise NotImplementedError
            
    def forward(self, input: paddle.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight).astype(paddle.float32)
            bias = self.bias.astype(paddle.float32) if self.bias is not None else None
        else:
            weight = self.weight.astype(paddle.float32)
            bias = self.bias.astype(paddle.float32) if self.bias is not None else None

        if not self.disable_act_quant and self.use_act_quant:
            input = self.act_quantizer(input.astype(paddle.float32))
            
        if self.bias is None or self.bias.ndim == 1:
            out = self.fwd_func(input.astype(paddle.float32), weight, bias, **self.fwd_kwargs)
        else:
            for i in range(bias.ndim):
                if bias.shape[i] == weight.shape[0]:
                    bias = paddle.transpose(bias, perm=[i, 0])
                    break
            out = self.fwd_func(input.astype(paddle.float32), weight, bias[:, 0], **self.fwd_kwargs)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def extra_repr(self):
        return "wbit={}, abit={}, disable_act_quant={}".format(
            self.weight_quantizer.n_bits,
            self.act_quantizer.n_bits,
            self.disable_act_quant,
        )
