import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import paddle
from PIL import Image
from tqdm import tqdm
from einops import rearrange
import math

from ppdiffusers.transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from ppdiffusers.image_processor import PipelineImageInput, VaeImageProcessor
from ppdiffusers.loaders import IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from ppdiffusers.models import AutoencoderKL, UNet3DConditionModel, UNetMotionModel
from ppdiffusers.models.lora import adjust_lora_scale_text_encoder
from ppdiffusers.models.unet_motion_model import MotionAdapter
from ppdiffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ppdiffusers.utils import USE_PEFT_BACKEND, BaseOutput, logging
from ppdiffusers.utils.paddle_utils import randn_tensor
from ppdiffusers.pipelines.pipeline_utils import DiffusionPipeline
from ppdiffusers.models.controlnet import ControlNetModel

from magicanimate.models.mutual_self_attention import ReferenceAttentionControl
from magicanimate.utils.util import get_tensor_interpolation_method
from magicanimate.pipelines.context import (
    get_context_scheduler
)
from magicanimate.models.base_model import BaseModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import paddle
        >>> from ppdiffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
        >>> from ppdiffusers.utils import export_to_gif

        >>> adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
        >>> pipe = AnimateDiffPipeline.from_pretrained("frankjoshua/toonyou_beta6", motion_adapter=adapter)
        >>> pipe.scheduler = DDIMScheduler(beta_schedule="linear", steps_offset=1, clip_sample=False)
        >>> output = pipe(prompt="A corgi walking in the park")
        >>> frames = output.frames[0]
        >>> export_to_gif(frames, "animation.gif")
        ```
"""


def tensor2vid(video: paddle.Tensor, processor, output_type="np"):
    # Based on:
    # https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78

    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].transpose([1, 0, 2, 3])
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    return outputs


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[paddle.Tensor, np.ndarray]


class AnimationPipeline(DiffusionPipeline, TextualInversionLoaderMixin, IPAdapterMixin, LoraLoaderMixin):
   
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["feature_extractor", "image_encoder"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        controlnet: ControlNetModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        feature_extractor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModelWithProjection = None,
    ):
        super().__init__()
        # unet = UNetMotionModel.from_unet2d(unet, motion_adapter)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.cali_data = {}

    def _encode_prompt(self, prompt, device, num_videos_per_prompt,
        do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        text_inputs = self.tokenizer(prompt, padding='max_length',
            max_length=self.tokenizer.model_max_length, truncation=True,
            return_tensors='pd')
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding='longest',
            return_tensors='pd').input_ids
        if tuple(untruncated_ids.shape)[-1] >= tuple(text_input_ids.shape)[-1
            ] and not paddle.equal_all(x=text_input_ids, y=untruncated_ids
            ).item():
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, 
                self.tokenizer.model_max_length - 1:-1])
            logger.warning(
                f'The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}'
                )
        if hasattr(self.text_encoder.config, 'use_attention_mask'
            ) and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None
        text_embeddings = self.text_encoder(text_input_ids.to(device),
            attention_mask=attention_mask)
        text_embeddings = text_embeddings[0]
        bs_embed, seq_len, _ = tuple(text_embeddings.shape)
        text_embeddings = text_embeddings.tile(repeat_times=[1,
            num_videos_per_prompt, 1])
        text_embeddings = text_embeddings.reshape([bs_embed *
            num_videos_per_prompt, seq_len, -1])
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [''] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}.'
                    )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`.'
                    )
            else:
                uncond_tokens = negative_prompt
            max_length = tuple(text_input_ids.shape)[-1]
            uncond_input = self.tokenizer(uncond_tokens, padding=
                'max_length', max_length=max_length, truncation=True,
                return_tensors='pd')
            if hasattr(self.text_encoder.config, 'use_attention_mask'
                ) and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to
                (device), attention_mask=attention_mask)
            uncond_embeddings = uncond_embeddings[0]
            seq_len = tuple(uncond_embeddings.shape)[1]
            uncond_embeddings = uncond_embeddings.tile(repeat_times=[1,
                num_videos_per_prompt, 1])
            uncond_embeddings = uncond_embeddings.reshape([batch_size *
                num_videos_per_prompt, seq_len, -1])
            text_embeddings = paddle.concat(x=[uncond_embeddings,
                text_embeddings])
        return text_embeddings

    def decode_latents(self, latents, rank, decoder_consistency=None):
        video_length = tuple(latents.shape)[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, 'b c f h w -> (b f) c h w')
        video = []
        for frame_idx in tqdm(range(tuple(latents.shape)[0]), disable=rank != 0
            ):
            if decoder_consistency is not None:
                video.append(decoder_consistency(latents[frame_idx:
                    frame_idx + 1]))
            else:
                video.append(self.vae.decode(latents[frame_idx:frame_idx + 
                    1]).sample)
        video = paddle.concat(x=video)
        video = rearrange(video, '(b f) c h w -> b c f h w', f=video_length)
        video = (video / 2 + 0.5).clip(min=0, max=1)
        video = video.cpu().astype(dtype='float32').numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).
            parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta
        accepts_generator = 'generator' in set(inspect.signature(self.
            scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(
                f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
                )
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f'`height` and `width` have to be divisible by 8 but are {height} and {width}.'
                )
        if callback_steps is None or callback_steps is not None and (not
            isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.'
                )

    def prepare_latents(self, batch_size, num_channels_latents,
        video_length, height, width, dtype, device, generator, latents=None,
        clip_length=16):
        shape = (batch_size, num_channels_latents, clip_length, height //
            self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
                )
        if latents is None:
            # rand_device = 'cpu' if device.type == 'mps' else device
            if isinstance(generator, list):
                latents = [paddle.randn(shape=shape, dtype=dtype) for i in
                    range(batch_size)]
                latents = paddle.concat(x=latents, axis=0).to(device)
            else:
                latents = paddle.randn(shape=shape, dtype=dtype).to(device)
            latents = latents.tile(repeat_times=[1, 1, video_length //
                clip_length, 1, 1])
        else:
            if tuple(latents.shape) != shape:
                raise ValueError(
                    f'Unexpected latents shape, got {tuple(latents.shape)}, expected {shape}'
                    )
            latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_condition(self, condition, num_videos_per_prompt, device,
        dtype, do_classifier_free_guidance):
        condition = paddle.to_tensor(data=condition.copy()).to(device=
            device, dtype=dtype) / 255.0
        condition = paddle.stack(x=[condition for _ in range(
            num_videos_per_prompt)], axis=0)
        condition = rearrange(condition, 'b f h w c -> (b f) c h w').clone()
        if do_classifier_free_guidance:
            condition = paddle.concat(x=[condition] * 2)
        return condition

    def next_step(self, model_output: paddle.Tensor, timestep: int, x:
        paddle.Tensor, eta=0.0, verbose=False):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print('timestep: ', timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps //
            self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep
            ] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        x_next = alpha_prod_t_next ** 0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    @paddle.no_grad()
    def images2latents(self, images, dtype):
        """
        Convert RGB image to VAE latents
        """
        device = 'gpu'
        images = paddle.to_tensor(data=images).astype(dtype='float32').to(dtype
            ) / 127.5 - 1
        images = rearrange(images, 'f h w c -> f c h w').to(device)
        latents = []
        for frame_idx in range(tuple(images.shape)[0]):
            latents.append(self.vae.encode(images[frame_idx:frame_idx + 1])
                ['latent_dist'].mean * 0.18215)
        latents = paddle.concat(x=latents)
        return latents

    @paddle.no_grad()
    def invert(self, image: paddle.Tensor, prompt, num_inference_steps=20,
        num_actual_inference_steps=10, eta=0.0, return_intermediates=False,
        **kwargs):
        """
        Adapted from: https://github.com/Yujun-Shi/DragDiffusion/blob/main/drag_pipeline.py#L440
        invert a real image into noise map with determinisc DDIM inversion
        """
        device = 'gpu'
        batch_size = tuple(image.shape)[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(shape=[len(prompt), -1, -1, -1])
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size
        text_input = self.tokenizer(prompt, padding='max_length',
            max_length=77, return_tensors='pd')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        print('input text embeddings :', tuple(text_embeddings.shape))
        latents = self.images2latents(image)
        print('latents shape: ', tuple(latents.shape))
        self.scheduler.set_timesteps(num_inference_steps)
        print('Valid timesteps: ', reversed(self.scheduler.timesteps))
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc
            ='DDIM Inversion')):
            if (num_actual_inference_steps is not None and i >=
                num_actual_inference_steps):
                continue
            model_inputs = latents
            model_inputs = rearrange(model_inputs, 'f c h w -> 1 c f h w')
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=
                text_embeddings).sample
            noise_pred = rearrange(noise_pred, 'b c f h w -> (b f) c h w')
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)
        if return_intermediates:
            return latents, latents_list
        return latents

    def interpolate_latents(self, latents: paddle.Tensor,
        interpolation_factor: int, device):
        if interpolation_factor < 2:
            return latents
        new_latents = paddle.zeros(shape=(tuple(latents.shape)[0], tuple(
            latents.shape)[1], (tuple(latents.shape)[2] - 1) *
            interpolation_factor + 1, tuple(latents.shape)[3], tuple(
            latents.shape)[4]), dtype=latents.dtype)
        org_video_length = tuple(latents.shape)[2]
        rate = [(i / interpolation_factor) for i in range(interpolation_factor)
            ][1:]
        new_index = 0
        v0 = None
        v1 = None
        for i0, i1 in zip(range(org_video_length), range(org_video_length)[1:]
            ):
            v0 = latents[:, :, (i0), :, :]
            v1 = latents[:, :, (i1), :, :]
            new_latents[:, :, (new_index), :, :] = v0
            new_index += 1
            for f in rate:
                v = get_tensor_interpolation_method()(v0.to(device=device),
                    v1.to(device=device), f)
                new_latents[:, :, (new_index), :, :] = v.to(latents.place)
                new_index += 1
        new_latents[:, :, (new_index), :, :] = v1
        new_index += 1
        return new_latents

    def select_controlnet_res_samples(self,
        controlnet_res_samples_cache_dict, context,
        do_classifier_free_guidance, b, f):
        _down_block_res_samples = []
        _mid_block_res_sample = []
        for i in np.concatenate(np.array(context)):
            _down_block_res_samples.append(controlnet_res_samples_cache_dict
                [i][0])
            _mid_block_res_sample.append(controlnet_res_samples_cache_dict[
                i][1])
        down_block_res_samples = [[] for _ in range(len(
            controlnet_res_samples_cache_dict[i][0]))]
        for res_t in _down_block_res_samples:
            for i, res in enumerate(res_t):
                down_block_res_samples[i].append(res)
        down_block_res_samples = [paddle.concat(x=res) for res in
            down_block_res_samples]
        mid_block_res_sample = paddle.concat(x=_mid_block_res_sample)
        b = b // 2 if do_classifier_free_guidance else b
        _down_block_res_samples = []
        for sample in down_block_res_samples:
            sample = rearrange(sample, '(b f) c h w -> b c f h w', b=b, f=f)
            if do_classifier_free_guidance:
                sample = sample.tile(repeat_times=[2, 1, 1, 1, 1])
            _down_block_res_samples.append(sample)
        down_block_res_samples = _down_block_res_samples
        mid_block_res_sample = rearrange(mid_block_res_sample,
            '(b f) c h w -> b c f h w', b=b, f=f)
        if do_classifier_free_guidance:
            mid_block_res_sample = mid_block_res_sample.tile(repeat_times=[
                2, 1, 1, 1, 1])
        return down_block_res_samples, mid_block_res_sample

    @paddle.no_grad()
    def __call__(self, prompt: Union[str, List[str]], video_length:
        Optional[int], height: Optional[int]=None, width: Optional[int]=
        None, num_inference_steps: int=50, guidance_scale: float=7.5,
        negative_prompt: Optional[Union[str, List[str]]]=None,
        num_videos_per_prompt: Optional[int]=1, eta: float=0.0, generator:
        Optional[Union[paddle.Generator, List[paddle.Generator]]]=None,
        latents: Optional[paddle.Tensor]=None, output_type: Optional[str]=
        'tensor', return_dict: bool=True, callback: Optional[Callable[[int,
        int, float], None]]=None, callback_steps: Optional[int]=1,
        controlnet_condition: list=None, controlnet_conditioning_scale:
        float=1.0, context_frames: int=16, context_stride: int=1,
        context_overlap: int=4, context_batch_size: int=1, context_schedule:
        str='uniform', init_latents: Optional[paddle.Tensor]=None,
        num_actual_inference_steps: Optional[int]=None, appearance_encoder=
        None, reference_control_writer=None, reference_control_reader=None,
        source_image: str=None, decoder_consistency=None, return_calidata=
        False, use_calidata=False, **kwargs):
        """
        New args:
        - controlnet_condition          : condition map (e.g., depth, canny, keypoints) for controlnet
        - controlnet_conditioning_scale : conditioning scale for controlnet
        - init_latents                  : initial latents to begin with (used along with invert())
        - num_actual_inference_steps    : number of actual inference steps (while total steps is num_inference_steps) 
        """
        paddle.device.cuda.empty_cache()
        controlnet = self.controlnet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(prompt, height, width, callback_steps)
        batch_size = 1
        if latents is not None:
            batch_size = tuple(latents.shape)[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)
        device = 'gpu'
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt,
                list) else [negative_prompt] * batch_size
        text_embeddings = self._encode_prompt(prompt, device,
            num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
            )
        text_embeddings = paddle.concat(x=[text_embeddings] *
            context_batch_size)
        reference_control_writer = ReferenceAttentionControl(appearance_encoder
            , do_classifier_free_guidance=True, mode='write', batch_size=
            context_batch_size)
        # print(self.unet)
        reference_control_reader = ReferenceAttentionControl(self.unet,
            do_classifier_free_guidance=True, mode='read', batch_size=
            context_batch_size)
        is_dist_initialized = kwargs.get('dist', False)
        rank = kwargs.get('rank', 0)
        world_size = kwargs.get('world_size', 1)
        assert num_videos_per_prompt == 1
        assert batch_size == 1
        control = self.prepare_condition(condition=controlnet_condition,
            device=device, dtype=controlnet.dtype, num_videos_per_prompt=
            num_videos_per_prompt, do_classifier_free_guidance=
            do_classifier_free_guidance)
        controlnet_uncond_images, controlnet_cond_images = control.chunk(chunks
            =2)
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        if init_latents is not None:
            latents = rearrange(init_latents, '(b f) c h w -> b c f h w', f
                =video_length)
        else:
            num_channels_latents = self.unet.model.in_channels
            latents = self.prepare_latents(batch_size *
                num_videos_per_prompt, num_channels_latents, video_length,
                height, width, text_embeddings.dtype, device, generator,
                latents)
        latents_dtype = latents.dtype
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        controlnet_text_embeddings = text_embeddings.repeat_interleave(repeats
            =video_length, axis=0)
        _, controlnet_text_embeddings_c = controlnet_text_embeddings.chunk(
            chunks=2)
        controlnet_res_samples_cache_dict = {i: None for i in range(
            video_length)}
        if num_actual_inference_steps is None:
            num_actual_inference_steps = num_inference_steps
        if isinstance(source_image, str):
            ref_image_latents = self.images2latents(np.array(Image.open(
                source_image).resize((width, height)))[(None), :],
                latents_dtype).cuda(blocking=True)
        elif isinstance(source_image, np.ndarray):
            ref_image_latents = self.images2latents(source_image[(None), :],
                latents_dtype).cuda(blocking=True)
        context_scheduler = get_context_scheduler(context_schedule)
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps),
            disable=rank != 0):
            if return_calidata:
                self.cali_data[t.item()] = latents
            if use_calidata:
                latents = self.cali_data[t.item()]
            if (num_actual_inference_steps is not None and i < 
                num_inference_steps - num_actual_inference_steps):
                continue
            noise_pred = paddle.zeros(shape=(tuple(latents.shape)[0] * (2 if
                do_classifier_free_guidance else 1), *tuple(latents.shape)[
                1:]), dtype=latents.dtype)
            counter = paddle.zeros(shape=(1, 1, tuple(latents.shape)[2], 1,
                1), dtype=latents.dtype)
            appearance_encoder(ref_image_latents.tile(repeat_times=[
                context_batch_size * (2 if do_classifier_free_guidance else
                1), 1, 1, 1]), t, encoder_hidden_states=text_embeddings,
                return_dict=False)
            context_queue = list(context_scheduler(0, num_inference_steps,
                tuple(latents.shape)[2], context_frames, context_stride, 0))
            num_context_batches = math.ceil(len(context_queue) /
                context_batch_size)
            for i in range(num_context_batches):
                context = context_queue[i * context_batch_size:(i + 1) *
                    context_batch_size]
                controlnet_latent_input = paddle.concat(x=[latents[:, :, (c
                    )] for c in context]).to(device)
                controlnet_latent_input = self.scheduler.scale_model_input(
                    controlnet_latent_input, t)
                b, c, f, h, w = tuple(controlnet_latent_input.shape)
                controlnet_latent_input = rearrange(controlnet_latent_input,
                    'b c f h w -> (b f) c h w')
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    controlnet_latent_input, t, encoder_hidden_states=
                    paddle.concat(x=[controlnet_text_embeddings_c[c] for c in
                    context]), controlnet_cond=paddle.concat(x=[
                    controlnet_cond_images[c] for c in context]),
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False)
                for j, k in enumerate(np.concatenate(np.array(context))):
                    controlnet_res_samples_cache_dict[k] = [sample[j:j + 1] for
                        sample in down_block_res_samples
                        ], mid_block_res_sample[j:j + 1]
            context_queue = list(context_scheduler(0, num_inference_steps,
                tuple(latents.shape)[2], context_frames, context_stride,
                context_overlap))
            num_context_batches = math.ceil(len(context_queue) /
                context_batch_size)
            global_context = []
            for i in range(num_context_batches):
                global_context.append(context_queue[i * context_batch_size:
                    (i + 1) * context_batch_size])
            for context in global_context[rank::world_size]:
                latent_model_input = paddle.concat(x=[latents[:, :, (c)] for
                    c in context]).to(device).tile(repeat_times=[2 if
                    do_classifier_free_guidance else 1, 1, 1, 1, 1])
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)
                b, c, f, h, w = tuple(latent_model_input.shape)
                down_block_res_samples, mid_block_res_sample = (self.
                    select_controlnet_res_samples(
                    controlnet_res_samples_cache_dict, context,
                    do_classifier_free_guidance, b, f))
                reference_control_reader.update(reference_control_writer)

                ##  process input shape
                # shape = down_block_res_samples[0].shape
                # value = down_block_res_samples[0]
                # num_frames = shape[2]
                # new_down_samples = value.transpose([0, 2, 1, 3, 4]).reshape([value.shape[0] * num_frames, -1] + value.shape[3:])
                # new_down_samples = [new_down_samples]
                
                # frames = mid_block_res_sample.shape[2]
                # new_mid_samples = mid_block_res_sample.transpose([0, 2, 1, 3, 4]).reshape([mid_block_res_sample.shape[0] * frames, -1] + mid_block_res_sample.shape[3:])

                pred = self.unet(latent_model_input, t,
                    encoder_hidden_states=text_embeddings[:b],
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False)[0]
                
                if t == 961:
                    for name, module in self.unet.named_sublayers():
                        if isinstance(module, BaseModel):
                            module.set_cac_migrate(False)
                reference_control_reader.clear()
                pred_uc, pred_c = pred.chunk(chunks=2)
                pred = paddle.concat(x=[pred_uc.unsqueeze(axis=0), pred_c.
                    unsqueeze(axis=0)])
                for j, c in enumerate(context):
                    noise_pred[:, :, (c)] = noise_pred[:, :, (c)] + pred[:, (j)
                        ]
                    counter[:, :, (c)] = counter[:, :, (c)] + 1
            if is_dist_initialized:
                noise_pred_gathered = [paddle.zeros_like(x=noise_pred) for
                    _ in range(world_size)]
                if rank == 0:
                    paddle.distributed.gather(tensor=noise_pred,
                        gather_list=noise_pred_gathered, dst=0)
                else:
                    paddle.distributed.gather(tensor=noise_pred,
                        gather_list=[], dst=0)
                paddle.distributed.barrier()
                if rank == 0:
                    for k in range(1, world_size):
                        for context in global_context[k::world_size]:
                            for j, c in enumerate(context):
                                noise_pred[:, :, (c)] = noise_pred[:, :, (c)
                                    ] + noise_pred_gathered[k][:, :, (c)]
                                counter[:, :, (c)] = counter[:, :, (c)] + 1
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = (noise_pred / counter
                    ).chunk(chunks=2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, **
                extra_step_kwargs).prev_sample
            if is_dist_initialized:
                paddle.distributed.broadcast(tensor=latents, src=0)
                paddle.distributed.barrier()
            reference_control_writer.clear()
        interpolation_factor = 1
        latents = self.interpolate_latents(latents, interpolation_factor,
            device)
        video = self.decode_latents(latents, rank, decoder_consistency=
            decoder_consistency)
        if is_dist_initialized:
            paddle.distributed.barrier()
        if output_type == 'tensor':
            video = paddle.to_tensor(data=video)
        if not return_dict:
            return video
        if return_calidata:
            return self.cali_data
        return AnimationPipelineOutput(videos=video)
