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

def create_fp8_map(signed=True, exponent_bits=5, precision_bits=2, total_bits=8):
    e = exponent_bits
    p = precision_bits
    has_sign = 1 if signed else 0
    assert e+p == total_bits-has_sign
    # the exponent is biased to 2^(e-1) -1 == 0
    evalues = []
    pvalues = []
    for i, val in enumerate(range(-((2**(exponent_bits-has_sign))), 2**(exponent_bits-has_sign), 1)):
        evalues.append(2**val)


    values = []
    lst = list(itertools.product([0, 1], repeat=precision_bits))
    #for ev in evalues:
    bias = 2**(exponent_bits-1)
    for evalue in range(2**(exponent_bits)):
        for bit_pattern in lst:
            value = (1 if evalue != 0 else 0)
            for i, pval in enumerate(list(bit_pattern)):
                value += pval*(2**-(i+1))
            if evalue == 0:
                # subnormals
                value = value*2**-(bias)
            else:
                # normals
                value = value*2**-(evalue-bias-1)
            values.append(value)
            if signed:
                values.append(-value)


    assert len(values) == 2**total_bits
    values.sort()
    if total_bits < 8:
        gap = 256 - len(values)
        for i in range(gap):
            values.append(0)
    values.sort()
    code /= code.max()

    return code


def quantize_nf4(x, blocksize):
    return quant_blockwise(x, None, blocksize=blocksize, quant_type="nf4")

def quantize_fp4(x, blocksize):
    return quant_blockwise(x, None, blocksize=blocksize, quant_type="fp4")

def dequantize_nf4(x, absmax, blocksize):
    return dequant_blockwise(x, None, absmax, blocksize=blocksize,  quant_type="nf4")

def dequantize_fp4(x, absmax, blocksize):
    return dequant_blockwise(x, None, absmax, blocksize=blocksize,  quant_type="fp4")


def quantize_8bit(x, code, blocksize):
    return quant_blockwise(x, code, blocksize=blocksize, quant_type="8bit")

def dequantize_8bit(x, code, absmax, blocksize):
    return dequant_blockwise(x, code, absmax, blocksize=blocksize, quant_type="8bit")

def double_quant(abs_max, code, blocksize, double_quant_type="dynamic"):
    offset = abs_max.mean()
    abs_max -= offset
    if code is None:
        if quant_type=="fp8":
            code = paddle.to_tensor(create_fp8_map(signed=True, exponent_bits=2, precision_bits=1, total_bits=4))
        else:
            code = paddle.to_tensor(create_dynamic_map())
    qabs_max, double_quant_scale = quantize_8bit(abs_max, code, blocksize)
    return qabs_max, double_quant_scale, offset, code

def double_dequant(qabs_max, offset, code, double_quant_scale, blocksize, double_quant_type="dynamic"):
    if code is None:
        if quant_type=="fp8":
            code = paddle.to_tensor(create_fp8_map(signed=True, exponent_bits=2, precision_bits=1, total_bits=4))
        else:
            code = paddle.to_tensor(create_dynamic_map())

    abs_max = dequantize_8bit(qabs_max, code, double_quant_scale, blocksize)
    abs_max += offset
    return abs_max

