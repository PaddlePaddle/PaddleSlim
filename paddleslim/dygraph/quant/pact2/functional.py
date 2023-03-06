import paddle


# straight-through estimation (STE) - this is the preferred mode for propagating outputs for quantization
# the backward gradients uses x (i.e. the input itself) and not y (the output)
# because the effect of y is completely detached during forward
def propagate_quant_ste(x, y):
    # this works functionally as STE, but exports an onnx graph containing
    # all the operators used to compute y as well
    # out = x + (y - x).detach()
    
    # torch version
    # this is another way of doing STE. in this case the operators used to generate y are skipped from onnx graph
    # out = x.clone()
    # out.data = y.data

    # paddle version
    out = x.clone()
    out.set_value(y)
    return out


# quantized through estimation - typically not used.
def propagate_quant_qte(x, y):
    return y


def round_func(x):
    y = paddle.round(x)
    return y


def round_g(x):
    return propagate_quant_ste(x, round_func(x))


def round_sym_func(x):
    rnd = (-0.5) * (x < 0).float() + (0.5) * (x >= 0).float()
    y = (x + rnd).int().float()
    return y


def round_sym_g(x):
    return propagate_quant_ste(x, round_sym_func(x))


def round_up_func(x):
    y = paddle.floor(x + 0.5)
    return y


def round_up_g(x):
    return propagate_quant_ste(x, round_up_func(x))


def round2_func(x):
    two = paddle.to_tensor(2, dtype=paddle.float32)
    y = paddle.pow(two, paddle.round(paddle.log2(x)))
    return y


def round2_g(x):
    return propagate_quant_ste(x, round2_func(x))


def ceil_func(x):
    y = paddle.ceil(x)
    return y


def ceil_g(x):
    return propagate_quant_ste(x, ceil_func(x))


def ceil2_func(x):
    two = paddle.to_tensor(2, dtype=paddle.float32)
    y = paddle.pow(two, paddle.ceil(paddle.log2(x)))
    return y


def ceil2_g(x):
    return propagate_quant_ste(x, ceil2_func(x))


def floor2_func(x):
    two = paddle.to_tensor(2, dtype=paddle.float32)
    y = paddle.pow(two, paddle.floor(paddle.log2(x)))
    return y


def floor2_g(x):
    return propagate_quant_ste(x, floor2_func(x))


def quantize_dequantize_func(x, scale_tensor, width_min: float, width_max: float, power2: bool, axis: int, round_type: str='round_up'):
    # clip values need ceil2 and scale values need floor2
    scale_tensor = floor2_func(scale_tensor) if power2 else scale_tensor
    x_scaled = (x * scale_tensor)

    # round
    if round_type == 'round_up': # typically for activations
        rand_val = 0.5
        x_scaled_round = paddle.floor(x_scaled + rand_val)
    elif round_type == 'round_sym': # typically for weights
        rand_val = (-0.5) * (x < 0).float() + (0.5) * (x >= 0).float()
        x_scaled_round = (x_scaled + rand_val).int().float()
    else:
        x_scaled_round = paddle.round(x_scaled)

    # invert the scale
    scale_inv = scale_tensor.pow(-1.0)
    # clamp
    x_clamp = paddle.clip(x_scaled_round, width_min, width_max)
    y = x_clamp * scale_inv
    return y, x_scaled_round


# quantization operation with STE gradients
def quantize_dequantize_g(x, *args, **kwargs):
    return propagate_quant_ste(x, quantize_dequantize_func(x, *args, **kwargs)[0])


def clamp_g(x, min, max, training, inplace=True, requires_grad=False):
    if x is None:
        return x
    #
    # in eval mode, torch.clamp can be used
    # the graph exported in eval mode will be simpler and have fixed constants that way.
    if training:
        if requires_grad:
            # torch's clamp doesn't currently work with min and max as tensors
            # TODO: replace with this, when torch clamp supports tensor arguments:
            # TODO:switch back to min/max if you want to lean the clip values by backprop
            zero_tensor = paddle.zeros_like(x.reshape([-1])[0])
            min = zero_tensor + min
            max = zero_tensor + max
            y = paddle.maximum(paddle.minimum(x, max), min)
        else:
            # clamp takes less memory - using it for now
            y = paddle.clip(x, min, max) if inplace else paddle.clip(x, min, max)
    else:
        y = paddle.clip(x, float(min), float(max)) if inplace else paddle.clip(x, float(min), float(max))

    return y

