import paddle
from .quant_layer import UniformAffineQuantizer, QuantModule
from .quant_block import BaseQuantBlock
from .set_act_quantize_params import set_act_quantize_params
from .set_weight_quantize_params import set_weight_quantize_params
from PIL import Image
import numpy as np


def resume_cali_model(animator, resume_param, args):
    print('warmup scale and zero point')
    animator.pipeline.unet.set_quant_state(weight_quant=True, act_quant=True)
    set_weight_quantize_params(animator.pipeline.unet)
    animator.pipeline.unet.set_quant_state(weight_quant=False, act_quant=True)
    set_act_quantize_params(resume_param['cali_warmup_file'], 1, 7.5,
        animator, steps=1, args=args)
    print(
        f"loading unet with scale and zero point from {resume_param['ckpt_path']}"
        )
    # sz_unet = paddle.load(path=str(resume_param['ckpt_path']))
    # act_quantize_state_dict = {k: v for k, v in sz_unet.items() if 
    #     'act_quantizer' in k}
    # animator.pipeline.unet.load_state_dict(act_quantize_state_dict, strict=
    #     False)
    # animator.pipeline.unet.set_quant_state(weight_quant=True, act_quant=True)
    for m in animator.pipeline.unet.sublayers():
        if isinstance(m, UniformAffineQuantizer):
            zero_data = m.zero_point.data
            delattr(m, 'zero_point')
            m.zero_point = zero_data
            delta_data = m.delta.data
            delattr(m, 'delta')
            m.delta = delta_data
