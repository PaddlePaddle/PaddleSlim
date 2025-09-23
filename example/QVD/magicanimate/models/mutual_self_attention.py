from typing import Any, Dict, Optional

import paddle
from einops import rearrange

from magicanimate.models.attention import (
    BasicTransformerBlock,
    TemporalBasicTransformerBlock,
)

from QDrop.quant.quant_model import QuantModel

def paddle_dfs(model: paddle.nn.Layer):
    result = [model]
    for child in model.children():
        result += paddle_dfs(child)
    return result


class ReferenceAttentionControl:
    def __init__(
        self,
        unet,
        mode="write",
        do_classifier_free_guidance=False,
        attention_auto_machine_weight=float("inf"),
        gn_auto_machine_weight=1.0,
        style_fidelity=1.0,
        reference_attn=True,
        reference_adain=False,
        fusion_blocks="midup",
        batch_size=1,
    ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            fusion_blocks,
            batch_size=batch_size,
        )

    def register_reference_hooks(
        self,
        mode,
        do_classifier_free_guidance,
        attention_auto_machine_weight,
        gn_auto_machine_weight,
        style_fidelity,
        reference_attn,
        reference_adain,
        dtype="float16",
        batch_size=1,
        num_images_per_prompt=1,
        fusion_blocks="midup",
    ):
        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance
        attention_auto_machine_weight = attention_auto_machine_weight
        gn_auto_machine_weight = gn_auto_machine_weight
        style_fidelity = style_fidelity
        reference_attn = reference_attn
        reference_adain = reference_adain
        fusion_blocks = fusion_blocks
        num_images_per_prompt = num_images_per_prompt
        dtype = dtype
        if do_classifier_free_guidance:
            uc_mask = paddle.to_tensor(
                data=[1] * batch_size * num_images_per_prompt * 16 + [0] * batch_size * num_images_per_prompt * 16,
                dtype="float32",
            ).astype(dtype="bool")
        else:
            uc_mask = paddle.to_tensor(data=[0] * batch_size * num_images_per_prompt * 2, dtype="float32").astype(
                dtype="bool"
            )

        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: paddle.Tensor,
            attention_mask: Optional[paddle.Tensor] = None,
            encoder_hidden_states: Optional[paddle.Tensor] = None,
            encoder_attention_mask: Optional[paddle.Tensor] = None,
            timestep: Optional[paddle.Tensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[paddle.Tensor] = None,
            video_length=None,
        ):

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                (norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp,) = self.norm1(
                    hidden_states,
                    timestep,
                    class_labels,
                    hidden_dtype=hidden_states.dtype,
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if MODE == "write":
                    self.bank.append(norm_hidden_states.clone())

                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                if MODE == "read":
                    bank_fea = [
                        rearrange(
                            d.unsqueeze(1).tile((1, video_length, 1, 1)).astype(norm_hidden_states.dtype),
                            "b t l c -> (b t) l c",
                        )
                        for d in self.bank
                    ]

                    modify_norm_hidden_states = paddle.concat(x=[norm_hidden_states] + bank_fea, axis=1)

                    hidden_states_uc = (
                        self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=modify_norm_hidden_states,
                            attention_mask=attention_mask,
                        )
                        + hidden_states
                    )
                    if do_classifier_free_guidance:
                        hidden_states_c = hidden_states_uc.clone()
                        _uc_mask = uc_mask.clone()
                        if hidden_states.shape[0] != _uc_mask.shape[0]:

                            _uc_mask = paddle.to_tensor(
                                data=[1] * (hidden_states.shape[0] // 2) + [0] * (hidden_states.shape[0] // 2),
                                dtype="float32",
                            ).astype(dtype="bool")

                        hidden_states_c[_uc_mask] = (
                            self.attn1(
                                norm_hidden_states[_uc_mask],
                                encoder_hidden_states=norm_hidden_states[_uc_mask],
                                attention_mask=attention_mask,
                            )
                            + hidden_states[_uc_mask]
                        )
                        hidden_states = hidden_states_c.clone()
                    else:
                        hidden_states = hidden_states_uc

                    # self.bank.clear()
                    if self.attn2 is not None:
                        # Cross-Attention
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm2(hidden_states)
                        )
                        hidden_states = (
                            self.attn2(
                                norm_hidden_states,
                                encoder_hidden_states=encoder_hidden_states,
                                attention_mask=attention_mask,
                            )
                            + hidden_states
                        )

                    # Feed-forward
                    hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

                    # Temporal-Attention
                    if self.unet_use_temporal_attention:
                        d = hidden_states.shape[1]
                        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
                        norm_hidden_states = (
                            self.norm_temp(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm_temp(hidden_states)
                        )
                        hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
                        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

                    return hidden_states

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        if self.reference_attn:
            if self.fusion_blocks == "midup":
                if isinstance(self.unet,QuantModel):
                    self.unet=self.unet.model
                attn_modules = [
                    module
                    for module in (paddle_dfs(self.unet.mid_block) + paddle_dfs(self.unet.up_blocks))
                    if isinstance(module, BasicTransformerBlock) or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                attn_modules = [
                    module
                    for module in paddle_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock) or isinstance(module, TemporalBasicTransformerBlock)
                ]
            attn_modules = sorted(attn_modules, key=lambda x: -x.norm1._normalized_shape[0])

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                if isinstance(module, BasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                if isinstance(module, TemporalBasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, TemporalBasicTransformerBlock
                    )

                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))

    def update(self, writer, dtype="float16"):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (paddle_dfs(self.unet.mid_block) + paddle_dfs(self.unet.up_blocks))
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in (paddle_dfs(writer.unet.mid_block) + paddle_dfs(writer.unet.up_blocks))
                    if isinstance(module, BasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module for module in paddle_dfs(self.unet) if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module for module in paddle_dfs(writer.unet) if isinstance(module, BasicTransformerBlock)
                ]
            reader_attn_modules = sorted(reader_attn_modules, key=lambda x: -x.norm1._normalized_shape[0])
            writer_attn_modules = sorted(writer_attn_modules, key=lambda x: -x.norm1._normalized_shape[0])
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                r.bank = [v.clone() for v in w.bank]
                # w.bank.clear()

    def clear(self):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (paddle_dfs(self.unet.mid_block) + paddle_dfs(self.unet.up_blocks))
                    if isinstance(module, BasicTransformerBlock) or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in paddle_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock) or isinstance(module, TemporalBasicTransformerBlock)
                ]
            reader_attn_modules = sorted(reader_attn_modules, key=lambda x: -x.norm1._normalized_shape[0])
            for r in reader_attn_modules:
                r.bank.clear()