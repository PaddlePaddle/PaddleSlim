
from quant_layers.conv import ChannelwiseBatchingQuantConv2d
from quant_layers.linear import PTQSLBatchingQuantLinear, PostGeluPTQSLBatchingQuantLinear
from quant_layers.matmul import PTQSLBatchingQuantMatMul, SoSPTQSLBatchingQuantMatMul

def get_module(module_type, *args, **kwargs):
    if module_type == 'qconv':
        kwargs.update(ptqsl_conv2d_kwargs)
        module = ChannelwiseBatchingQuantConv2d(*args, **kwargs, w_bit=
            w_bit['qconv'], a_bit=32)
    elif 'qlinear' in module_type:
        kwargs.update(ptqsl_linear_kwargs)
        if module_type == 'qlinear_qkv':
            kwargs['n_V'] *= 3
            module = PTQSLBatchingQuantLinear(*args, **kwargs, w_bit=w_bit[module_type], a_bit=a_bit[module_type])
        elif module_type == 'qlinear_MLP_2':
            if no_postgelu:
                module = PTQSLBatchingQuantLinear(*args, **kwargs, w_bit=w_bit[module_type], a_bit=a_bit[module_type])
            else:
                module = PostGeluPTQSLBatchingQuantLinear(*args, **kwargs, w_bit=w_bit[module_type], a_bit=a_bit[module_type])
        elif module_type == 'qlinear_classifier':
            kwargs['n_V'] = 1
            module = PTQSLBatchingQuantLinear(*args, **kwargs, w_bit=w_bit[module_type], a_bit=a_bit[module_type])
        else:
            module = PTQSLBatchingQuantLinear(*args, **kwargs, w_bit=w_bit[module_type], a_bit=a_bit[module_type])
    elif 'qmatmul' in module_type:
        kwargs.update(ptqsl_matmul_kwargs)
        if module_type == 'qmatmul_qk':
            module = PTQSLBatchingQuantMatMul(*args, **kwargs, A_bit=A_bit[module_type], B_bit=B_bit[module_type])
        elif module_type == 'qmatmul_scorev':
            if no_softmax:
                module = PTQSLBatchingQuantMatMul(*args, **kwargs, A_bit=A_bit[module_type], B_bit=B_bit[module_type])
            else:
                module = SoSPTQSLBatchingQuantMatMul(*args, **kwargs, A_bit=A_bit[module_type], B_bit=B_bit[module_type])
    return module
