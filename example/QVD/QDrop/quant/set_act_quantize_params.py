from .quant_layer import QuantModule
from .data_utils import save_inp_oup_data
import paddle
from .quant_block import BaseQuantBlock, QuantInflatedConv3d, QuantLoRACompatibleLinear
from .quant_model import QuantModel
from typing import Union
import numpy as np
from PIL import Image

def get_init(model, block, cali_data, wq, aq, batch_size, input_prob: bool=
    False, keep_gpu: bool=True):
    cached_inps, cached_outs = save_inp_oup_data(model, block, cali_data,
        wq, aq, batch_size, input_prob=input_prob, keep_gpu=keep_gpu)
    return cached_inps, cached_outs

def set_weight_quantize_params(model):
    print(f'set_weight_quantize_params')
    for name, module in model.named_sublayers():
        if isinstance(module, QuantModule):
            module.weight_quantizer.set_inited(False)
            """caculate the step size and zero point for weight quantizer"""
            module.weight_quantizer(module.weight)
            module.weight_quantizer.set_inited(True)

def weight_get_quant_state(order, act_quant):
    if not act_quant:
        return True, False
    if order == 'before':
        weight_quant, act_quant = True, True
    elif order == 'after':
        weight_quant, act_quant = True, False
    elif order == 'together':
        weight_quant, act_quant = True, True
    else:
        raise NotImplementedError
    return weight_quant, act_quant

def save_quantized_weight(model):
    for module in model.sublayers():
        if isinstance(module, QuantModule):
            module.weight.set_value(module.weight_quantizer(module.weight))

def set_act_quantize_params(
    cali_data_file,
    seed, guidance_scale,
    module: Union[QuantModel, QuantModule, BaseQuantBlock],
    awq: bool = False,
    order: str = "before",
    batch_size: int = 256,
    steps=25,
    args=None

):
    module.pipeline.unet.set_quant_state(False, True)

    for t in module.pipeline.unet.sublayers():
        if isinstance(t, (QuantModule, BaseQuantBlock)):
            if not isinstance(t, (QuantInflatedConv3d, QuantLoRACompatibleLinear)):
                t.act_quantizer.set_inited(False)

    module.set_quanting(True)
    """set or init step size and zero point in the activation quantizer"""
    with open(cali_data_file,'r') as f:
        lines = f.readlines()
        for line in lines:
            ref_img_p, motion_seq_p = line.split(' ')
            motion_seq_p = motion_seq_p.rstrip()
            cali_ref_img = np.array(Image.open(ref_img_p))
            animation = module(cali_ref_img, motion_seq_p, seed, steps, guidance_scale, return_calidata=False,use_calidata=args.use_calidata)
    module.set_quanting(False)

    module.pipeline.unet.set_quant_state(True, True)

    for t in module.pipeline.unet.sublayers():
        if isinstance(t, (QuantModule, BaseQuantBlock)):
            if not isinstance(t, (QuantInflatedConv3d, QuantLoRACompatibleLinear)):
                t.act_quantizer.set_inited(True)

    if not args.resume_sz:
        print("saveing sz checkpoint")
        state_dict = module.pipeline.unet.state_dict()
        act_quantize_state_dict = {k: v for k, v in state_dict.items() if "act_quantizer" in k}
        paddle.save(act_quantize_state_dict, args.sz_ckpt_output_path)
    paddle.device.cuda.empty_cache()

def act_get_quant_state(order, awq):
    if order == "before":
        weight_quant, act_quant = False, True
    elif order == "after":
        weight_quant, act_quant = awq, True
    elif order == "together":
        weight_quant, act_quant = True, True
    else:
        raise NotImplementedError
    return weight_quant, act_quant
