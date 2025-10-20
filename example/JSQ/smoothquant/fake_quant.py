from functools import partial

import paddle
from loguru import logger

############################## 相关utils函数，如下 ##############################

def _Tensor_max(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret

setattr(paddle.Tensor, "_max", _Tensor_max)

def _Tensor_view(self, *args, **kwargs):
    if args:
        if len(args)==1 and isinstance(args[0], (tuple, list, str)):
            return paddle.view(self, args[0])
        else:
            return paddle.view(self, list(args))
    elif kwargs:
        return paddle.view(self, shape_or_dtype = list(kwargs.values())[0])

setattr(paddle.Tensor, 'view', _Tensor_view)
############################## 相关utils函数，如上 ##############################



@paddle.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # 获取每个channel的最大绝对值作为scale
    scales = w.abs().max(keepdim=True, axis=-1)[0]
    
    # 计算量化的最大值
    q_max = 2 ** (n_bits - 1) - 1
    
    # 关键修复：确保类型匹配
    # 方法1：将q_max转换为与scales相同的float类型
    scales = paddle.clip(scales, min=1e-05) / float(q_max)
    
    # 量化权重：除以scale，四舍五入，再乘回scale
    w = paddle.round(w / scales) * scales
    
    return w


@paddle.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    scales = w.abs()._max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clip_(min=1e-05).divide_(y=paddle.to_tensor(q_max))
    w.divide_(y=paddle.to_tensor(scales)).round_().multiply_(y=paddle.to_tensor(scales))
    return w


@paddle.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = tuple(t.shape)
    t.view(-1, t_shape[-1])
    scales = (
        t.abs().max(keepdim=True, axis=-1),
        t.abs().argmax(keepdim=True, axis=-1),
    )[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clip_(min=1e-05).divide_(y=paddle.to_tensor(q_max))
    t.divide_(y=paddle.to_tensor(scales)).round_().multiply_(y=paddle.to_tensor(scales))
    return t


@paddle.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = tuple(t.shape)
    t.view(-1, t_shape[-1])
    scales = t.abs()._max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clip_(min=1e-05).divide_(y=paddle.to_tensor(q_max))
    t.divide_(y=paddle.to_tensor(scales)).round_().multiply_(y=paddle.to_tensor(scales))
    return t


class W8A8Linear(paddle.nn.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
        nbits=6,
    ):
        super().__init__()
        self.nbits = nbits
        self.in_features = in_features
        self.out_features = out_features
        out_0 = paddle.randn(
            shape=[self.out_features, self.in_features], dtype="float16"
        )
        out_0.stop_gradient = not False
        self.register_buffer(name="weight", tensor=out_0)
        if bias:
            out_1 = paddle.zeros(shape=(1, self.out_features), dtype="float16")
            out_1.stop_gradient = not False
            self.register_buffer(name="bias", tensor=out_1)
        else:
            self.register_buffer(name="bias", tensor=None)
        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=self.nbits
            )
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=self.nbits
            )
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")
        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        """Not Support auto convert *.to, please judge whether it is Pytorch API and convert by yourself"""
        super(W8A8Linear, self).to(*args, **kwargs)
        """Not Support auto convert *.to, please judge whether it is Pytorch API and convert by yourself"""
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            """Not Support auto convert *.to, please judge whether it is Pytorch API and convert by yourself"""
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @paddle.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        # wufazidong
        # y = torch.functional.F.linear(q_x, self.weight, self.bias)
        y = paddle.nn.functional.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_output=False,
        nbits=8,
    ):
        assert isinstance(module, paddle.nn.Linear)
        # import pdb; pdb.set_trace()
        new_module = W8A8Linear(
            module.weight.shape[0],
            module.weight.shape[0],
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            nbits=nbits,
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=nbits
            )
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=nbits
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"