import paddle
from numpy import dtype
from quant_layers.matmul import MinMaxQuantMatMul, PTQSLBatchingQuantMatMul, PTQSLQuantMatMul, SoSPTQSLBatchingQuantMatMul, SoSPTQSLQuantMatMul
from quant_layers.linear import MinMaxQuantLinear, PTQSLBatchingQuantLinear, PostGeluPTQSLBatchingQuantLinear, PostGeluPTQSLQuantLinear


def quantize_int_weight(module):
    """
    get weight of type 'uint8' of a quantized module.
    Bias are not quantized and you can use raw bias.
    """
    assert hasattr(module, 'weight'), f'module {module} does not have weight'
    assert module.w_bit == 8, f"module {module}'s weight is quantized with {module.w_bit} bits"
    w_int = (module.weight / module.w_interval).round_().clip_(min=-module.
        w_qmax, max=module.w_qmax - 1)
    w_int = w_int.cpu().detach().to('int8')
    return w_int


def dequantize_int_weight(module, w_int):
    """
    Make sure it's the same module that generates w_int
    """
    w_sim = module.w_interval.cpu() * w_int.astype(dtype='float32')
    return w_sim


def quantize_matmul_input(input, interval, qmax, n_G, n_V, n_H, crb_groups,
    crb_rows, crb_cols):
    """
    quantize input matrix of matmul operation, with respect to sublayerwise padding settings
    """
    pad_groups = crb_groups * n_G - tuple(input.shape)[1]
    pad_rows = crb_rows * n_V - tuple(input.shape)[2]
    pad_cols = crb_cols * n_H - tuple(input.shape)[3]
    x = paddle.nn.functional.pad(x=input, pad=[0, pad_cols, 0, pad_rows, 0,
        pad_groups], pad_from_left_axis=False)
    x = x.view(-1, n_G, crb_groups, n_V, crb_rows, n_H, crb_cols)
    x = (x / interval).round_().clip(min=-qmax, max=qmax - 1)
    x = x.view(-1, n_G * crb_groups, n_V * crb_rows, n_H * crb_cols)
    x = x[:, :tuple(x.shape)[1] - pad_groups, :tuple(x.shape)[2] - pad_rows,
        :tuple(x.shape)[3] - pad_cols]
    return x


def quantize_int_activation(module, input):
    """
    Quantize current inputs into uint8 and store them as an attribute of the module.

    The function is a pre-forward hook that need to be manually added to the calibrated model.
    You need to manipulate the cached data before feeding another batch of pictures.
    Currently only support int8. (For twin quantization, we use uint8)

    For twin quantization:
    - For softmax, the MSB being 1 means using large interval, while MSB being 0 means using small interval.
    - For post-GELU, the MSB serves as sign bit. We use 1 for positive values and 0 for negative values.
    """
    if isinstance(module, PostGeluPTQSLQuantLinear) or isinstance(module,
        PostGeluPTQSLBatchingQuantLinear):
        assert module.a_bit == 8, f"module {module}'s activation is quantized with {module.a_bit} bits"
        x = input[0]
        int_input_pos = (x / module.a_interval).round_().clip_(min=0, max=
            module.a_qmax - 1)
        int_input_pos = int_input_pos.detach().to('uint8') + 128
        int_input_neg = (x / module.a_neg_interval).round_().clip_(min=-
            module.a_qmax + 1, max=0).abs()
        int_input_neg = int_input_neg.detach().to('uint8')
        int_input = (int_input_pos + int_input_neg).cpu()
        module.int_input = [int_input]
    elif isinstance(module, MinMaxQuantLinear):
        assert module.a_bit == 8, f"module {module}'s activation is quantized with {module.a_bit} bits"
        x = input[0]
        int_input = (x / module.a_interval).round_().clip_(min=-module.
            a_qmax, max=module.a_qmax - 1)
        int_input = int_input.cpu().detach().to('int8')
        module.int_input = [int_input]
    elif isinstance(module, SoSPTQSLQuantMatMul) or isinstance(module,
        SoSPTQSLBatchingQuantMatMul):
        assert module.A_bit == 8, f"module {module}'s matrix A is quantized with {module.A_bit} bits"
        assert module.B_bit == 8, f"module {module}'s matrix B is quantized with {module.B_bit} bits"
        A, B = input[0], input[1]
        A_high = (A.clip(min=module.split, max=1) * (module.A_qmax - 1)
            ).round_().clip_(min=0, max=module.A_qmax - 1)
        A_high = A_high.detach().to('uint8') + 128
        A_low = (A.clip(min=0, max=module.split) / module.A_interval).round_(
            ).clip_(min=0, max=module.A_qmax - 1)
        A_low = A_low.detach().to('uint8')
        A_int = (A_high + A_low).cpu()
        B_int = quantize_matmul_input(B, module.B_interval, module.B_qmax,
            module.n_G_B, module.n_V_B, module.n_H_B, module.crb_groups_B,
            module.crb_rows_B, module.crb_cols_B)
        B_int = B_int.cpu().detach().to('int8')
        module.int_input = [A_int, B_int]
    elif isinstance(module, PTQSLQuantMatMul) or isinstance(module,
        PTQSLBatchingQuantMatMul):
        assert module.A_bit == 8, f"module {module}'s matrix A is quantized with {module.A_bit} bits"
        assert module.B_bit == 8, f"module {module}'s matrix B is quantized with {module.B_bit} bits"
        A, B = input[0], input[1]
        A_int = quantize_matmul_input(A, module.A_interval, module.A_qmax,
            module.n_G_A, module.n_V_A, module.n_H_A, module.crb_groups_A,
            module.crb_rows_A, module.crb_cols_A)
        A_int = A_int.cpu().detach().to('int8')
        B_int = quantize_matmul_input(B, module.B_interval, module.B_qmax,
            module.n_G_B, module.n_V_B, module.n_H_B, module.crb_groups_B,
            module.crb_rows_B, module.crb_cols_B)
        B_int = B_int.cpu().detach().to('int8')
        module.int_input = [A_int, B_int]


def get_model_int_weight(wrapped_modules):
    """
    Get quantized weights (in int8) of a model.

    Return:
        A dict, with modules' names as keys, and int weights as values.
    """
    int_weights = {}
    for name, m in wrapped_modules.items():
        try:
            int_weights[name] = quantize_int_weight(m)
        except:
            pass
    return int_weights
