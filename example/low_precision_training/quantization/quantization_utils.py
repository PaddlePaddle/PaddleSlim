import math
import paddle

def parse_qtype(qtype_str):
    qtype = qtype_str.rstrip('1234567890')
    bitwidth = qtype_str[len(qtype):]
    if bitwidth == '':
        bitwidth = 0
    elif bitwidth.isdigit():
        bitwidth = int(bitwidth)
    else:
        raise TypeError(f'dtype error for {qtype_str}')
    return qtype, bitwidth

def stochastic_round(tensor):
    return paddle.sign(tensor) * paddle.floor(paddle.abs(tensor) + paddle.rand(tensor.shape))

def custom_round(tensor, stochastic):
    if stochastic:
        return paddle.sign(tensor) * paddle.floor(paddle.abs(tensor) + paddle.rand(tensor.shape))
    else:
        return paddle.round(tensor)

def quantize_by_mse(tensor, bitwidth, quant_type='int', quant_granularity='per-layer', axis=None, max_iter=50):
    """quantize each layer by minimizing MSE"""
    if quant_type == 'full':
        return paddle.to_tensor(1, dtype='float32'), tensor
    assert int(bitwidth) == bitwidth, 'bitwidth must be integer'
    if bitwidth >= 2:
        if quant_type == 'int':
            MAX_VAL = math.pow(2., bitwidth - 1) - 1.
            MIN_VAL = -MAX_VAL
        elif quant_type == 'uint':
            MIN_VAL = 0
            MAX_VAL = math.pow(2, bitwidth) - 1
        else:
            raise RuntimeError(f"invalid quant_type: {quant_type}")

        if quant_granularity == 'per-layer':
            size = tensor.shape
            flatten_tensor = paddle.reshape(tensor, [-1])
            alpha_old = -1.0
            alpha = paddle.max(paddle.abs(flatten_tensor)) / MAX_VAL
            eps = 1e-6
            for it in range(max_iter):
                flatten_tensor_q = paddle.clip(paddle.round(flatten_tensor / alpha), MIN_VAL, MAX_VAL)
                alpha_old = alpha
                alpha = paddle.dot(flatten_tensor, flatten_tensor_q) / paddle.dot(flatten_tensor_q, flatten_tensor_q)
                if paddle.abs((alpha - alpha_old) / alpha) >= eps:
                    break
            quantized_tensor = paddle.reshape(flatten_tensor_q * alpha, size)

        elif quant_granularity == 'per-channel':
            size = tensor.shape
            flatten_tensor = paddle.reshape(tensor, [size[0], -1])
            alpha = paddle.max(paddle.abs(flatten_tensor), axis=1) / MAX_VAL
            alpha_old = paddle.ones_like(alpha) * -1.0
            eps = 1e-6
            for it in range(max_iter):
                flatten_tensor_q = paddle.clip(paddle.round(flatten_tensor / alpha[:, None]), MIN_VAL, MAX_VAL)
                alpha_old = alpha
                alpha = paddle.sum(flatten_tensor * flatten_tensor_q, axis=1) / paddle.sum(flatten_tensor_q * flatten_tensor_q, axis=1)
                if paddle.sum(paddle.abs((alpha - alpha_old) / alpha) >= eps).sum() > 0:  # 需要询问
                    break
            quantized_tensor = paddle.reshape(flatten_tensor_q * alpha[:, None], size)

        else:
            raise RuntimeError(f"invalid quant_granularity: {quant_granularity}")

    elif bitwidth == 1:
        assert quant_type == 'int'
        MIN_VAL = -1.
        MAX_VAL = 1.

        if quant_granularity == 'per-layer':
            alpha = paddle.mean(paddle.abs(tensor))
            tensor_q = paddle.sign(tensor)
            quantized_tensor = tensor_q * alpha

        elif quant_granularity == 'per-channel':
            size = tensor.shape
            flatten_tensor = paddle.reshape(tensor, [size[0], -1])
            alpha = paddle.mean(paddle.abs(flatten_tensor), axis=1)
            flatten_tensor_q = paddle.sign(flatten_tensor)
            quantized_tensor = paddle.reshape(flatten_tensor_q * alpha[:, None], size)

        else:
            raise RuntimeError(f"invalid quant_granularity: {quant_granularity}")

    else:
        raise RuntimeError(f"invalid bitwidth: {bitwidth}")
    
    return alpha, quantized_tensor
