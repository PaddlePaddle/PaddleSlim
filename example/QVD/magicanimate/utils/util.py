import sys
sys.path.append('/Users/yifuding/Downloads/paddle_project/utils')
import os
import paddle
import imageio
import numpy as np
from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange
from typing import Any, BinaryIO, List, Optional, Tuple, Union
from itertools import repeat
from types import FunctionType
import math

def make_grid(tensor: Union[paddle.Tensor, List[paddle.Tensor]], nrow: int=
    8, padding: int=2, normalize: bool=False, value_range: Optional[Tuple[
    int, int]]=None, scale_each: bool=False, pad_value: float=0.0
    ) ->paddle.Tensor:
    if isinstance(tensor, list):
        tensor = paddle.stack(x=tensor, axis=0)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(axis=0)
    if tensor.dim() == 3:
        if tensor.shape[0] == 1:
            tensor = paddle.concat(x=(tensor, tensor, tensor), axis=0)
        tensor = tensor.unsqueeze(axis=0)
    if tensor.dim() == 4 and tensor.shape[1] == 1:
        tensor = paddle.concat(x=(tensor, tensor, tensor), axis=1)
    if normalize is True:
        tensor = tensor.clone()
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError(
                'value_range has to be a tuple (min, max) if specified. min and max are numbers'
                )

        def norm_ip(img, low, high):
            img.clip_(min=low, max=high)
            img.subtract_(y=paddle.to_tensor(low)).divide_(y=paddle.
                to_tensor(max(high - low, 1e-05)))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))
        if scale_each is True:
            for t in tensor:
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)
    if not isinstance(tensor, paddle.Tensor):
        raise TypeError('tensor should be of type torch.Tensor')
    if tensor.shape[0] == 1:
        return tensor.squeeze(axis=0)
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] +
        padding)
    num_channels = tensor.shape[1]
    grid = paddle.full(shape=(num_channels, height * ymaps + padding, width *
        xmaps + padding), fill_value=pad_value, dtype=tensor.dtype)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            start_0 = (grid.shape[1] + y * height + padding if y * height +
                padding < 0 else y * height + padding)
            start_1 = (paddle.slice(grid, [1], [start_0], [start_0 + height -
                padding]).shape[2] + x * width + padding if x * width +
                padding < 0 else x * width + padding)
            paddle.assign(tensor[k], output=paddle.slice(paddle.slice(grid,
                [1], [start_0], [start_0 + height - padding]), [2], [
                start_1], [start_1 + width - padding]))
            k = k + 1
    return grid

def save_videos_grid(videos: paddle.Tensor, path: str, rescale=False,
    n_rows=6, fps=25, args=None, quanting=True, timestep=25):
    videos = rearrange(videos, 'b c t h w -> t b c h w')
    outputs = []
    img_prefix = path.split('/')[-1].split('.')[0]
    for i, x in enumerate(videos):
        x = make_grid(x, nrow=n_rows)
        # import pdb; pdb.set_trace()
        x = paddle.transpose(x, perm=(1, 0, 2))
        x = paddle.transpose(x, perm=(0, 2, 1)).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)
        if not quanting:
            img_name = f'{img_prefix}_{str(i).zfill(7)}.png'
            img_path = os.path.join(args.save_video_imgs_dir, img_name)
            imageio.imwrite(img_path, x)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def save_images_grid(images: paddle.Tensor, path: str):
    assert tuple(images.shape)[2] == 1
    images = images.squeeze(axis=2)
    grid = torchvision.utils.make_grid(images)
    grid = (grid * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(grid).save(path)


@paddle.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer([''], padding='max_length',
        max_length=pipeline.tokenizer.model_max_length, return_tensors='pt')
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(
        pipeline.place))[0]
    text_input = pipeline.tokenizer([prompt], padding='max_length',
        max_length=pipeline.tokenizer.model_max_length, truncation=True,
        return_tensors='pt')
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(
        pipeline.place))[0]
    context = paddle.concat(x=[uncond_embeddings, text_embeddings])
    return context


def next_step(model_output: Union[paddle.Tensor, np.ndarray], timestep: int,
    sample: Union[paddle.Tensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(timestep - ddim_scheduler.config.
        num_train_timesteps // ddim_scheduler.num_inference_steps, 999
        ), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep
        ] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output
        ) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = (alpha_prod_t_next ** 0.5 * next_original_sample +
        next_sample_direction)
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)['sample']
    return noise_pred


@paddle.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(chunks=2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings,
            pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@paddle.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps,
    prompt=''):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent,
        num_inv_steps, prompt)
    return ddim_latents


def video2images(path, step=4, length=16, start=0):
    reader = imageio.get_reader(path)
    frames = []
    for frame in reader:
        frames.append(np.array(frame))
    frames = frames[start::step][:length]
    return frames


def images2video(video, path, fps=8):
    imageio.mimsave(path, video, fps=fps)
    return


tensor_interpolation = None

def get_tensor_interpolation_method():
    return tensor_interpolation


def set_tensor_interpolation_method(is_slerp):
    global tensor_interpolation
    tensor_interpolation = slerp if is_slerp else linear


def linear(v1, v2, t):
    return (1.0 - t) * v1 + t * v2


def slerp(v0: paddle.Tensor, v1: paddle.Tensor, t: float, DOT_THRESHOLD:
    float=0.9995) ->paddle.Tensor:
    u0 = v0 / v0.norm()
    u1 = v1 / v1.norm()
    dot = (u0 * u1).sum()
    if dot.abs() > DOT_THRESHOLD:
        return (1.0 - t) * v0 + t * v1
    omega = dot.acos()
    return (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1
        ) / omega.sin()
