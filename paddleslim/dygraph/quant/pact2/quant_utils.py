import math
import paddle
from functional import *

def even_round_tensor(tensor_in):
    # rounding towards even. banker's rounding (Unbiased rounding)
    # this is what numpy and pytorch does.
    tensor = tensor_in.round()
    return tensor


def symmetric_round_tensor(tensor_in):
    # rounding away from zero. This is also reasonbably unbiased, although not the best.
    rnd = (-0.5) * (tensor_in < 0).float() + (+0.5) * (tensor_in >= 0).float()
    tensor = tensor_in + rnd
    tensor = tensor.int().float()
    return tensor


def upward_round_tensor(tensor_in):
    # round towards +infinity. done in typical fixed point hardware.
    tensor = (tensor_in + 0.5).floor().int().float()
    return tensor


def compute_tensor_scale(tensor, mn, mx, bitwidth, power2_scaling, force_data_type=None):
    if force_data_type == 'signed':
        signed = True
    elif force_data_type == 'unsigned':
        signed = False
    else:
        signed = min(mn, mx) < 0
    abs_range = max(abs(mn), abs(mx))
    valid_range = not (math.isinf(abs_range) or math.isnan(abs_range) or abs_range == 0)
    
    extrabits = 1 if signed else 0
    bitwidth_range = math.pow(2.0, bitwidth - extrabits)
    clamp_limits = (-bitwidth_range, bitwidth_range - 1) if signed else (0, bitwidth_range - 1)

    tensor_bits = math.ceil(math.log2(abs_range)) if valid_range else 0.0
    tensor_range_pow2 = math.pow(2.0, tensor_bits) if valid_range else 1.0
    tensor_range_to_use = tensor_range_pow2 if power2_scaling else abs_range
    tensor_scale = (bitwidth_range / tensor_range_to_use) if valid_range else 1.0
    tensor_scale = min(tensor_scale, 65536)
    # if ((tensor is not None) and (not power2_scaling)):
    #     # Weight scaling as in: "DSConv: Efficient Convolution Operator"
    #     # https://arxiv.org/pdf/1901.01928.pdf
    #     quantized_tensor = torch.round(tensor * tensor_scale)
    #     den = torch.dot(tensor.view(-1), quantized_tensor.view(-1))
    #     num = torch.dot(quantized_tensor.view(-1), quantized_tensor.view(-1))
    #     den = max(den, 1e-6)
    #     tensor_scale = (num / den)
    #     tensor_scale = min(tensor_scale, 65536)
    return tensor_scale, clamp_limits, signed


def clamp_weight_simple(merged_weight, clamp_ratio, clamp_value):
    # a simple clamp - this may not be suitable for all models
    clamped_weight = merged_weight.clamp(-clamp_value, clamp_value)
    return clamped_weight


def clamp_weight_soft(weight, clamp_ratio, clamp_value):
    weight_max = weight.abs().max()
    weight_median = weight.abs().median()
    if (weight_max > clamp_value) and (weight_max > (weight_median * clamp_ratio)):
        # weight = torch.tanh(weight/clamp_value)*(clamp_value)
        weight = paddle.tanh(weight/clamp_value)*(clamp_value)
    #
    return weight


def clamp_weight_ratio(merged_weight, clamp_ratio, clamp_value):
    # an intlligent clamp - look at the statistics and then clamp
    weight_max = merged_weight.abs().max()
    weight_median = merged_weight.abs().median()
    if (weight_max > clamp_value) and (weight_max > (weight_median * clamp_ratio)):
        # weight_max = torch.min(weight_max, weight_median * clamp_ratio)
        weight_max = paddle.min(weight_max, weight_median * clamp_ratio)
        weight_max2 = ceil2_g(weight_max)
        scale_max2 = 128.0 / weight_max2
        # minimum 1 - using slightly higher margin to ensure quantization aware training
        # will not immediately cause the weights to move to the next scale range
        clamp_margin = 8
        clamp_max2 = (weight_max2 * scale_max2 - clamp_margin) / scale_max2
        clamped_weight = merged_weight.clamp(-float(clamp_max2), float(clamp_max2))
    else:
        clamped_weight = merged_weight
    #
    return clamped_weight


def constrain_weight(weight, clamp_ratio=16.0, clamp_value=15.0):
    '''
    for a mild constraining: use clamp_ratio=32.0, clamp_value=31.0
    for aggressive constraining use: clamp_ratio=16.0, clamp_value=15.0
    '''
    # weight = clamp_weight_simple(weight, clamp_ratio, clamp_value)
    # weight = clamp_weight_soft(weight, clamp_ratio, clamp_value)
    weight = clamp_weight_ratio(weight, clamp_ratio, clamp_value)
    return weight
