import paddle
import paddlenlp
# from smoothquant.modeling_chatglm import RMSNorm

############################## 相关utils函数，如下 ##############################

def _Tensor_view(self, *args, **kwargs):
    if args:
        if len(args)==1 and isinstance(args[0], (tuple, list, str)):
            return paddle.view(self, args[0])
        else:
            return paddle.view(self, list(args))
    elif kwargs:
        return paddle.view(self, shape_or_dtype = list(kwargs.values())[0])

setattr(paddle.Tensor, 'view', _Tensor_view)
############################## 相关utils函数，如上 ##############################



@paddle.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert (
        isinstance(ln, paddle.nn.LayerNorm)
        # wufazidong
        # or isinstance(ln, transformers.models.llama.modeling_llama.LlamaRMSNorm)
        or isinstance(ln, paddlenlp.transformers.llama.modeling.LlamaRMSNorm)
        # or isinstance(ln, RMSNorm)
        or "RMSNorm" in ln.__class__.__name__
    )
    # for fc in fcs:
    #     assert isinstance(fc, paddle.nn.Linear)
    #     assert ln.weight.size == fc.in_features == act_scales.size
    device, dtype = fcs[0].weight.place, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    # import pdb; pdb.set_trace()
    weight_scales = paddle.concat(
        x=[
            (
                fc.weight.transpose([1,0]).abs().max(keepdim=True, axis=0),
                fc.weight.transpose([1,0]).abs().argmax(keepdim=True, axis=0),
            )[0]
            for fc in fcs
        ],
        axis=0,
    )
    weight_scales = (weight_scales.max(axis=0), weight_scales.argmax(axis=0))[0].clip(
        min=1e-05
    )
    
    scales = (
        (act_scales.pow(y=alpha) / weight_scales.pow(y=1 - alpha))
        .clip(min=1e-05)
        .to(device)
        .to(dtype)
    )
    ln.weight.divide_(y=paddle.to_tensor(scales))
    if hasattr(ln, "bias"):
        ln.bias.divide_(y=paddle.to_tensor(scales))
    for fc in fcs:
        fc.weight.transpose_([1,0]).multiply_(y=paddle.to_tensor(scales.view(1, -1))).transpose_([1,0])


@paddle.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    for name, module in model.named_sublayers(include_self=True):
        # wufazidong
        # if isinstance(module, transformers.models.opt.modeling_opt.OPTDecoderLayer):
        if isinstance(module, paddlenlp.transformers.OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + ".fc1"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        # wufazidong
        # elif isinstance(
        #     module, transformers.models.llama.modeling_llama.LlamaDecoderLayer
        # ):
        elif isinstance(
            module, paddlenlp.transformers.llama.modeling_llama.LlamaDecoderLayer
        ):
            attn_ln = module.input_layernorm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            if "layer" in name:
                qkv_input_scales = scales[name + ".self_attn.q_proj"]
            else:
                qkv_input_scales = scales["self_attn.q_proj"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
            ffn_ln = module.post_attention_layernorm
            fc1 = [module.mlp.gate_proj, module.mlp.up_proj]
            if "layer" in name:
                fc1_input_scales = scales[name + ".mlp.up_proj"]
            else:
                fc1_input_scales = scales["mlp.up_proj"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        # wufazidong
        # elif isinstance(module, transformers.models.bloom.modeling_bloom.BloomBlock):
        elif isinstance(module, paddlenlp.transformers.bloom.modeling_bloom.BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)


@paddle.no_grad()
def smooth_layer(name, module, scales, alpha=0.5):
    # wufazidong
    # if isinstance(module, transformers.models.llama.modeling_llama.LlamaDecoderLayer):
    if isinstance(module, paddlenlp.transformers.llama.modeling.LlamaDecoderLayer):
    # 注意：这里假设PaddleNLP中有对应的LlamaDecoderLayer类，实际使用时需要确认PaddleNLP的版本和API。
        attn_ln = module.input_layernorm
        qkv = [
            module.self_attn.q_proj,
            module.self_attn.k_proj,
            module.self_attn.v_proj,
        ]
        qkv_input_scales = scales[name + ".self_attn.q_proj"]
        smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
        ffn_ln = module.post_attention_layernorm
        fc1 = [module.mlp.gate_proj, module.mlp.up_proj]
        fc1_input_scales = scales[name + ".mlp.gate_proj"]
        smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
    elif "FalconDecoderLayer" in module.__class__.__name__:
        attn_ln = module.input_layernorm
        qkv_fc1 = [module.self_attention.query_key_value, module.mlp.dense_h_to_4h]
        qkv_input_scales = scales[name + ".self_attention.query_key_value"]
        smooth_ln_fcs(attn_ln, qkv_fc1, qkv_input_scales, alpha)
    elif "GLMBlock" in module.__class__.__name__:
        attn_ln = module.input_layernorm
        qkv = [module.self_attention.query_key_value]
        qkv_input_scales = scales[name + ".self_attention.query_key_value"]
        smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
        ffn_ln = module.post_attention_layernorm
        fc1 = [module.mlp.dense_h_to_4h]
        fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
        smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
    else:
        raise TypeError(f"未添加的Decoder类{module}")