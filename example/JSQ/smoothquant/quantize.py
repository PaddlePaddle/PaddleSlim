import paddle
import paddlenlp
from smoothquant.fake_quant import W8A8Linear
# from smoothquant.modeling_chatglm import GLMBlock


@paddle.no_grad()
def quantize_model(
    model, weight_quant="per_tensor", act_quant="per_tensor", quantize_bmm_input=True
):
    for name, m in model.named_sublayers(include_self=True):
        # wufazidong
        # if isinstance(m, transformers.models.opt.modeling_opt.OPTDecoderLayer):
        if isinstance(m, paddlenlp.transformers.opt.modeling_opt.OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(
                m.fc1, weight_quant=weight_quant, act_quant=act_quant
            )
            m.fc2 = W8A8Linear.from_float(
                m.fc2, weight_quant=weight_quant, act_quant=act_quant
            )
        # wufazidong
        # elif isinstance(m, transformers.models.opt.modeling_opt.OPTAttention):
        elif isinstance(m, paddlenlp.transformers.opt.modeling_opt.OPTAttention):
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.out_proj = W8A8Linear.from_float(
                m.out_proj, weight_quant=weight_quant, act_quant=act_quant
            )
        # wufazidong
        # elif isinstance(m, transformers.models.llama.modeling_llama.LlamaAttention):
        elif isinstance(m, paddlenlp.transformers.LlamaAttention):
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant
            )
        # wufazidong
        # elif isinstance(m, transformers.models.llama.modeling_llama.LlamaMLP):
        # 注意：这里假设LlamaMLP在PaddleNLP的LlamaForCausalLM模型中有类似的定义，
        # 实际使用时需要确认PaddleNLP中是否有对应的类或模块，并可能需要调整导入路径。
        elif isinstance(m, paddlenlp.transformers.LlamaForCausalLM.LlamaMLP):
            m.gate_proj = W8A8Linear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.down_proj = W8A8Linear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.up_proj = W8A8Linear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant
            )
    return model


@paddle.no_grad()
def quantize_layer(
    module,
    nbits,
    weight_quant="per_channel",
    act_quant="per_token",
    quantize_bmm_input=True,
):
    for name, m in module.named_sublayers(include_self=True):
        # wufazidong
        # if isinstance(m, transformers.models.opt.modeling_opt.OPTDecoderLayer):
        if isinstance(m, paddlenlp.transformers.llama.modeling.LlamaAttention):
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                nbits=nbits,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                nbits=nbits,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                nbits=nbits,
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant, nbits=nbits
            )
        # wufazidong
        # elif isinstance(m, transformers.models.llama.modeling_llama.LlamaDecoderLayer):
        elif isinstance(m, paddlenlp.transformers.llama.modeling.LlamaDecoderLayer):
            m.mlp.gate_proj = W8A8Linear.from_float(
                m.mlp.gate_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                nbits=nbits,
            )
            m.mlp.down_proj = W8A8Linear.from_float(
                m.mlp.down_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                nbits=nbits,
            )
            m.mlp.up_proj = W8A8Linear.from_float(
                m.mlp.up_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                nbits=nbits,
            )
        elif "FalconDecoderLayer" in m.__class__.__name__:
            m.self_attention.query_key_value = W8A8Linear.from_float(
                m.self_attention.query_key_value,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                nbits=nbits,
            )
            m.self_attention.dense = W8A8Linear.from_float(
                m.self_attention.dense,
                weight_quant=weight_quant,
                act_quant=act_quant,
                nbits=nbits,
            )
            m.mlp.dense_h_to_4h = W8A8Linear.from_float(
                m.mlp.dense_h_to_4h,
                weight_quant=weight_quant,
                act_quant=act_quant,
                nbits=nbits,
            )
            m.mlp.dense_4h_to_h = W8A8Linear.from_float(
                m.mlp.dense_4h_to_h,
                weight_quant=weight_quant,
                act_quant=act_quant,
                nbits=nbits,
            )
        # elif isinstance(m, GLMBlock) or "GLMBlock" in m.__class__.__name__:
        #     m.self_attention.query_key_value = W8A8Linear.from_float(
        #         m.self_attention.query_key_value,
        #         weight_quant=weight_quant,
        #         act_quant=act_quant,
        #         quantize_output=quantize_bmm_input,
        #         nbits=nbits,
        #     )
        #     m.self_attention.dense = W8A8Linear.from_float(
        #         m.self_attention.dense,
        #         weight_quant=weight_quant,
        #         act_quant=act_quant,
        #         nbits=nbits,
        #     )
        #     m.mlp.dense_h_to_4h = W8A8Linear.from_float(
        #         m.mlp.dense_h_to_4h,
        #         weight_quant=weight_quant,
        #         act_quant=act_quant,
        #         nbits=nbits,
        #     )
        #     m.mlp.dense_4h_to_h = W8A8Linear.from_float(
        #         m.mlp.dense_4h_to_h,
        #         weight_quant=weight_quant,
        #         act_quant=act_quant,
        #         nbits=nbits,
        #     )
    return module