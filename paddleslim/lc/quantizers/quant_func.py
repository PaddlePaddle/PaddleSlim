import paddle
from custom_setup_ops import quant_blockwise, dequant_blockwise

def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    """
    Creates the dynamic quantiztion map.

    The dynamic data type is made up of a dynamic exponent and
    fraction. As the exponent increase from 0 to -7 the number
    of bits available for the fraction shrinks.

    This is a generalization of the dynamic type where a certain
    number of the bits and be reserved for the linear quantization
    region (the fraction). n determines the maximum number of
    exponent bits.

    For more details see
    (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561]
    """

    data = []
    # these are additional items that come from the case
    # where all the exponent bits are zero and no
    # indicator bit is present
    non_sign_bits = total_bits - (1 if signed else 1)
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        fraction_items = int((2 ** (i + non_sign_bits - max_exponent_bits) + 1 if signed else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1))
        boundaries = paddle.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    if additional_items > 0:
        boundaries = paddle.linspace(0.1, 1, additional_items + 1)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    data.append(0)
    data.append(1.0)

    assert len(data) == 2**total_bits

    gap = 256 - len(data)
    for i in range(gap):
        data.append(0)

    data.sort()
    return data

def quantize_nf4(x, blocksize):
    return quant_blockwise(x, None, blocksize=blocksize, quant_type="nf4")

def quantize_fp4(x, blocksize):
    return quant_blockwise(x, None, blocksize=blocksize, quant_type="fp4")

def dequantize_nf4(x, absmax, blocksize):
    return dequant_blockwise(x, None, absmax, blocksize=blocksize,  quant_type="nf4")

def dequantize_fp4(x, absmax, blocksize):
    return dequant_blockwise(x, None, absmax, blocksize=blocksize,  quant_type="fp4")


def quantize_fp8(x, code, blocksize):
    return quant_blockwise(x, code, blocksize=blocksize, quant_type="8bit")

def dequantize_fp8(x, code, absmax, blocksize):
    return dequant_blockwise(x, code, absmax, blocksize=blocksize, quant_type="8bit")

def double_quant(abs_max, code, blocksize):
    offset = abs_max.mean()
    abs_max -= offset
    if code is None:
        code = paddle.to_tensor(create_dynamic_map())
    qabs_max, double_quant_scale = quantize_fp8(abs_max, code, blocksize)
    return qabs_max, double_quant_scale, offset, code

def double_dequant(qabs_max, offset, code, double_quant_scale, blocksize):
    if code is None:
        code = paddle.to_tensor(create_dynamic_map())

    abs_max = dequantize_fp8(qabs_max, code, double_quant_scale, blocksize)
    abs_max += offset
    return abs_max

