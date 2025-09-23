import sys
import paddle


def round_ste(x: paddle.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def fake_quantize_per_tensor_affine(x, scale, zero_point, quant_min, quant_max
    ):
    x_int = round_ste(x / scale) + zero_point
    x_quant = paddle.clip(x=x_int, min=quant_min, max=quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def quantize_per_channel_affine(x, scale, zero_point, ch_axis, quant_min,
    quant_max):
    new_shape = [1] * len(tuple(x.shape))
    new_shape[ch_axis] = tuple(x.shape)[ch_axis]
    scale = scale.reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = paddle.clip(x=x_int, min=quant_min, max=quant_max)
    return x_quant


def dequantize_per_channel_affine(x, scale, zero_point, ch_axis, quant_min,
    quant_max):
    new_shape = [1] * len(tuple(x.shape))
    new_shape[ch_axis] = tuple(x.shape)[ch_axis]
    scale = scale.reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)
    x_dequant = (x - zero_point) * scale
    return x_dequant


def quantize_per_tensor_affine(x, scale, zero_point, quant_min, quant_max):
    x_int = round_ste(x / scale) + zero_point
    x_quant = paddle.clip(x=x_int, min=quant_min, max=quant_max)
    return x_quant


def dequantize_per_tensor_affine(x, scale, zero_point, quant_min, quant_max):
    x_dequant = (x - zero_point) * scale
    return x_dequant


def fake_quantize_per_channel_affine(x, scale, zero_point, ch_axis,
    quant_min, quant_max):
    new_shape = [1] * len(tuple(x.shape))
    new_shape[ch_axis] = tuple(x.shape)[ch_axis]
    scale = scale.reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = paddle.clip(x=x_int, min=quant_min, max=quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant
