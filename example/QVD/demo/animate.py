
import argparse
import argparse
import datetime
import inspect
import os
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict

import paddle

from ppdiffusers.transformers import CLIPTextModel, CLIPTokenizer
from ppdiffusers.models import UNet3DConditionModel,  ControlNetModel, AutoencoderKL
from ppdiffusers.schedulers.scheduling_ddim import DDIMScheduler
from ppdiffusers.models.unet_2d_condition import UNet2DConditionModel

from magicanimate.utils.videoreader import VideoReader
from magicanimate.models.mutual_self_attention import ReferenceAttentionControl
from magicanimate.pipelines.pipeline_animation import AnimationPipeline
from magicanimate.utils.util import save_videos_grid

class MagicAnimate():
    def __init__(self, args, config="configs/prompts/animation.yaml") -> None:
        print("Initializing MagicAnimate Pipeline...")
        *_, func_args = inspect.getargvalues(inspect.currentframe())
        func_args = dict(func_args)
        self.args = args
        self.device = 'gpu'
        config  = OmegaConf.load(config)
        self.quanting = False
        
        inference_config = OmegaConf.load(config.inference_config)
            
        motion_module = config.motion_module
       
        print(self.args.save_dir)
        print(args.save_video_imgs_dir)

        ### >>> create animation pipeline >>> ###
        tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_path, subfolder="tokenizer", from_diffusers=True)
        # text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_path, subfolder="text_encoder", from_diffusers=True)
        text_encoder = CLIPTextModel.from_pretrained('runwayml/stable-diffusion-v1-5', 
                                  subfolder='text_encoder', 
                                  from_hf_hub=True)
        if config.pretrained_unet_path:
            unet = UNet3DConditionModel.from_pretrained(config.pretrained_unet_path, unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs),from_diffusers=True)
        else:
            unet = UNet3DConditionModel.from_pretrained(config.pretrained_model_path, subfolder="unet-3d", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs), from_diffusers=True,
                                                        ignore_mismatched_sizes=True)
        # else:
        #     if config.pretrained_unet_path:
        #         unet = UNet3DConditionModel.get_model_without_weight(config.pretrained_unet_path, unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
        #     else:
        #         unet = UNet3DConditionModel.get_model_without_weight(config.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
        
        self.appearance_encoder = UNet2DConditionModel.from_pretrained(config.pretrained_appearance_encoder_path, subfolder="appearance_encoder", from_diffusers=True)
        self.reference_control_writer = ReferenceAttentionControl(self.appearance_encoder, do_classifier_free_guidance=True, mode='write', fusion_blocks=config.fusion_blocks)
        self.reference_control_reader = ReferenceAttentionControl(unet, do_classifier_free_guidance=True, mode='read', fusion_blocks=config.fusion_blocks)
        if config.pretrained_vae_path is not None:
            vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path, from_diffusers=True)
        else:
            vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, subfolder="vae", from_diffusers=True)

        ### Load controlnet
        controlnet  = ControlNetModel.from_pretrained(config.pretrained_controlnet_path, from_diffusers=True)

        # vae.to(paddle.float16)
        # unet.to(paddle.float16)
        # text_encoder.to(paddle.float16)
        # controlnet.to(paddle.float16)
        # self.appearance_encoder.to(paddle.float16)

        
        from ppdiffusers import is_ppxformers_available
        if is_ppxformers_available():   
            print("Enabling XFormers Memory Efficient Attention!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            print("=========================XFormers Memory Efficient Attention is not available, please install ppxformers to enable it.")
        unet.enable_xformers_memory_efficient_attention()
        self.appearance_encoder.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()

        print(inference_config.noise_scheduler_kwargs)
        self.pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            # NOTE: UniPCMultistepScheduler
        )

        # 1. unet ckpt
        # 1.1 motion module
        # if not self.args.resume_sz:

        # motion_module_state_dict = paddle.load(motion_module)
        # if "global_step" in motion_module_state_dict: func_args.update({"global_step": motion_module_state_dict["global_step"]})
        # motion_module_state_dict = motion_module_state_dict['state_dict'] if 'state_dict' in motion_module_state_dict else motion_module_state_dict
        # try:
        #     # extra steps for self-trained models
        #     state_dict = OrderedDict()
        #     for key in motion_module_state_dict.keys():
        #         if key.startswith("module."):
        #             _key = key.split("module.")[-1]
        #             state_dict[_key] = motion_module_state_dict[key]
        #         else:
        #             state_dict[key] = motion_module_state_dict[key]
        #     motion_module_state_dict = state_dict
        #     del state_dict
        #     missing, unexpected = self.pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
        #     assert len(unexpected) == 0
        # except:
        #     _tmp_ = OrderedDict()
        #     for key in motion_module_state_dict.keys():
        #         if "motion_modules" in key:
        #             if key.startswith("unet."):
        #                 _key = key.split('unet.')[-1]
        #                 _tmp_[_key] = motion_module_state_dict[key]
        #             else:
        #                 _tmp_[key] = motion_module_state_dict[key]
        #     missing, unexpected = unet.load_state_dict(_tmp_, strict=False)
        #     assert len(unexpected) == 0
        #     del _tmp_
        # del motion_module_state_dict

        # self.pipeline.to("cuda")
        self.L = config.L
        
        print("Initialization Done!")


    def convert_model_to_fp32(self, model):
        for param in model.parameters():
            param.set_value(param.astype(paddle.float32))
            if param.grad is not None:
                param._grad_ivar().set_value(param._grad_ivar().astype(paddle.float32))

    def set_quanting(self, quanting):
        self.quanting = quanting

    def __call__(self, source_image, motion_sequence, random_seed, step, guidance_scale, size=512, use_calidata=False, return_calidata=False):
        return self.forward(source_image, motion_sequence, random_seed, step, guidance_scale, size, use_calidata=use_calidata, return_calidata=return_calidata)

    def get_cali_data(self, source_image, motion_sequence, random_seed, step, guidance_scale, size=512):
        return self.pipeline_forward(source_image, motion_sequence, random_seed, step, guidance_scale, size, return_calidata=True)

    def pipeline_forward(self, source_image, motion_sequence, random_seed, step, guidance_scale, size=512, return_calidata=False, use_calidata=False):
        prompt = ""
        n_prompt = ""

        random_seed = int(random_seed)
        step = int(step)
        guidance_scale = float(guidance_scale)
        samples_per_video = []

        # Manually set random seed for reproduction
        if random_seed != -1:
            paddle.seed(random_seed)
        else:
            paddle.seed()

        if motion_sequence.endswith('.mp4'):
            control = VideoReader(motion_sequence).read()
            if control[0].shape[0] != size:
                control = [np.array(Image.fromarray(c).resize((size, size))) for c in control]
            control = np.array(control)

        if source_image.shape[0] != size:
            source_image = np.array(Image.fromarray(source_image).resize((size, size)))
        H, W, C = source_image.shape

        init_latents = None
        original_length = control.shape[0]
        if control.shape[0] % self.L > 0:
            control = np.pad(control, ((0, self.L - control.shape[0] % self.L), (0, 0), (0, 0), (0, 0)), mode='edge')

        generator = paddle.seed(random_seed)

        video_per_timestep = self.pipeline(
            prompt,
            negative_prompt=n_prompt,
            num_inference_steps=step,
            guidance_scale=guidance_scale,
            width=W,
            height=H,
            video_length=len(control),
            controlnet_condition=control,
            init_latents=init_latents,
            generator=generator,
            appearance_encoder=self.appearance_encoder,
            reference_control_writer=self.reference_control_writer,
            reference_control_reader=self.reference_control_reader,
            source_image=source_image,
            return_calidata=return_calidata,
            use_calidata=use_calidata
        )
        if return_calidata:
            return video_per_timestep
        return video_per_timestep.videos, original_length

    def forward(self, source_image, motion_sequence, random_seed, step, guidance_scale, size=512, return_calidata=False, use_calidata=False):
        sample, original_length = self.pipeline_forward(
            source_image,
            motion_sequence,
            random_seed,
            step,
            guidance_scale,
            size=size,
            return_calidata=return_calidata,
            use_calidata=use_calidata
        )

        samples_per_video = []
        samples_per_video.append(sample[:, :, :original_length])

        samples_per_video = paddle.concat(samples_per_video, axis=0)

        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        savedir = self.args.save_dir
        vid_name = motion_sequence.split("/")[-1]
        if self.quanting:
            animation_path = f"{savedir}/quant/{vid_name}"
        else:
            animation_path = f"{savedir}/{vid_name}"

        os.makedirs(savedir, exist_ok=True)
        os.makedirs(self.args.save_video_imgs_dir, exist_ok=True)
        save_videos_grid(samples_per_video, animation_path, args=self.args, quanting=self.quanting)

        return animation_path


            