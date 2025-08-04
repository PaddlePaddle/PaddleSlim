import paddle
import os
import sys

import argparse
import numpy as np
from PIL import Image
from typing import Union
from tqdm import tqdm
from QDrop.quant.utils import resume_cali_model
from demo.animate import MagicAnimate
from QDrop.quant import block_reconstruction, layer_reconstruction, BaseQuantBlock, QuantModule, QuantModel, set_weight_quantize_params, set_act_quantize_params


def quant_model(args, model):
    # print(model)
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise,
        'scale_method': args.init_wmode, 'symmetric': True}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False,
        'scale_method': args.init_amode, 'leaf_param': True, 'prob': args.
        prob, 'per_batch': True, 'symmetric': True}
    qnn = QuantModel(model=model, weight_quant_params=wq_params,
        act_quant_params=aq_params)
    # qnn.cuda(blocking=True)
    # print('QuantModel created!!!!!')
    # print(qnn)
    qnn.eval()
    if not args.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()
    qnn.disable_network_output_quantization()
    cali_data = None
    assert args.wwq is True
    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.
        weight, b_range=(args.b_start, args.b_end), warmup=args.warmup,
        opt_mode='mse', wwq=args.wwq, waq=args.waq, order=args.order,
        act_quant=args.act_quant, lr=args.lr, input_prob=args.input_prob,
        keep_gpu=not args.keep_cpu)
    if args.act_quant and args.order == 'before' and args.awq is False:
        """Case 2"""
        set_act_quantize_params(qnn, cali_data=cali_data, awq=args.awq,
            order=args.order)
    if not args.use_adaround:
        print('setting')
        qnn.set_quant_state(weight_quant=True, act_quant=args.act_quant)
        return qnn
    else:

        def set_weight_act_quantize_params(module):
            if isinstance(module, QuantModule):
                layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                block_reconstruction(qnn, module, **kwargs)
            else:
                raise NotImplementedError

        def recon_model(model: paddle.nn.Layer):
            """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
            """
            for name, module in model.named_children():
                if isinstance(module, QuantModule):
                    print('Reconstruction for layer {}'.format(name))
                    set_weight_act_quantize_params(module)
                elif isinstance(module, BaseQuantBlock):
                    print('Reconstruction for block {}'.format(name))
                    set_weight_act_quantize_params(module)
                else:
                    recon_model(module)
        recon_model(qnn)
        if args.act_quant and args.order == 'after' and args.waq is False:
            """Case 1"""
            set_act_quantize_params(qnn, cali_data=cali_data, awq=args.awq,
                order=args.order)
        qnn.set_quant_state(weight_quant=True, act_quant=args.act_quant)
        return qnn


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected')


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f'--{k}', default=v, type=v_type)


def get_calidata(animator, args, reference_image, motion_path, seed, steps,
    guidance_scale):
    animator.pipeline.unet.set_quant_state(weight_quant=False, act_quant=False)
    cali_data = animator.get_cali_data(reference_image, motion_path, seed,
        steps, guidance_scale)
    animator.pipeline.unet.set_quant_state(weight_quant=True, act_quant=
        args.act_quant)
    return cali_data


def main(reference_image_path, motion_sequence_path, cali_image,
    cali_motion, seed, steps, guidance_scale, args):
    animator = MagicAnimate(args)
    reference_image = np.array(Image.open(args.image))
    for name,layer in animator.pipeline.unet.named_sublayers():
        print(name, type(layer))
    animator.pipeline.unet = quant_model(args=args, model=animator.pipeline
        .unet)
    # print(animator.pipeline.unet)
    args.save_video_imgs_dir = os.path.join(args.save_video_imgs_dir, args.
        run_name)
    if args.resume_sz:
        resume_param = {'ckpt_path': args.sz_ckpt_path, 'cali_warmup_file':
            args.warmup_file, 'warmup_motion': args.warmup_motion}
        resume_cali_model(animator, resume_param, args)
    else:
        """init weight quantizer"""
        print(f'start qunatize weight using {args.init_wmode}')
        set_weight_quantize_params(animator.pipeline.unet)
        print('finished qunatize weight')
        if args.act_quant:
            print(f'start act quant using {args.init_amode}')
            set_act_quantize_params(args.cali_file, seed, guidance_scale,
                animator, steps=steps, args=args)
            print('act quant finished')
    animation = animator(reference_image, args.motion, seed, steps,
        guidance_scale)
    return animation


if __name__ == '__main__':
    defaults = dict(clip_denoised=True, num_samples=10000, batch_size=500,
        use_ddim=True, model_path=
        '/mnt/disk2/charles/template/QVD/model/ptqdiffusionmodel/imagenet64_uncond_100M_1500K.pt'
        , image_size=64, num_channels=128, num_res_blocks=3, learn_sigma=
        True, diffusion_steps=4000, timestep_respacing=250, noise_schedule=
        'cosine')
    parser = argparse.ArgumentParser(description=
        'Animate images using MagicAnimate.')
    parser.add_argument('--run_name', default='pred_debug', type=str, help=
        '保存图片的目录前缀')
    parser.add_argument('--use_calidata', action='store_true', default=
        False, help='是否收集并使用全精度的校验集')
    parser.add_argument('--sz_ckpt_path', default=
        '/mnt/disk2/charles/template/QVD/sz_ckpt/quick_32f_64n_w8a8_cali_data_32f_1n__per_batch_update_quantize_range_abs_avg_w:minmax_a:mse.pth'
        , type=str, help='保存了scale 和 zero_p的完整权重路径，是个QuantModel')
    parser.add_argument('--resume_sz', action='store_true', default=False,
        help='是否load scale和zero_p')
    parser.add_argument('--sz_ckpt_output_path', default=
        '/mnt/disk2/charles/template/QVD/sz_ckpt/sz2.pth', type=str, help=
        'scale和zero_p保存路径')
    parser.add_argument('--save_dir', default=
        '/mnt/disk2/charles/template/QVD/demo/debug', type=str, help='生成视频保存路径'
        )
    parser.add_argument('--image', default=
        '/mnt/disk1/datasets/TED/ted_test_png/test/jr5mTwfFh00#014054#014208.mp4/0000000.png'
        , help='Path to the reference image')
    parser.add_argument('--motion', default=
        '/mnt/disk1/datasets/TED/ted_test_png/motion_32/jr5mTwfFh00#014054#014208.mp4'
        , help='Path to the motion sequence video')
    parser.add_argument('--warmup_motion', default=
        '/mnt/disk2/charles/template/QVD/inputs/applications/driving/densepose/demo4_16.mp4'
        , help='Path to the motion sequence video')
    parser.add_argument('--warmup_file', default=
        '/mnt/disk2/charles/template/QVD/cali_data/cali_data_warmup_16f.txt',
        help='Path to the motion sequence video')
    parser.add_argument('--save_video_imgs_dir', default=
        '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-platcv/tsl/code/outputs/TED/magic_animate'
        , help='Path to the motion sequence video')
    parser.add_argument('--ref_img_dir', default=
        '/mnt/disk1/datasets/TED/ted_test_png/test', help=
        'Path to the reference image')
    parser.add_argument('--motion_dir', default=
        '/mnt/disk1/datasets/TED/ted_test_png/motion_32', help=
        'Path to the motion sequence video')
    parser.add_argument('--quick_test_refImg_dir', default=
        '/mnt/disk1/datasets/TED/ted_test_png/quick_32f_64n_frames_gt',
        help='Path to the motion sequence video')
    parser.add_argument('--quick_test_motion_dir', default=
        '/mnt/disk1/datasets/TED/ted_test_png/quick_32f_64n', help=
        'Path to the motion sequence video')
    parser.add_argument('--device', default='cuda:0', help=
        'Path to the motion sequence video')
    parser.add_argument('--cali_file', default=
        '/mnt/disk2/charles/template/QVD/cali_data/cali_data_32f_1n.txt',
        help='Path to 检验数据目录文件')
    parser.add_argument('--cali_image', default=
        '/mnt/disk1/datasets/TED/ted_test_png/test/zpjxElfNpks#010561#010716.mp4/0000000.png'
        , help='Path to the reference image')
    parser.add_argument('--cali_motion', default=
        '/mnt/disk2/charles/template/QVD/inputs/applications/driving/densepose/demo4_16.mp4'
        , help='Path to the motion sequence video')
    parser.add_argument('--seed', type=int, default=1, help=
        'Random seed (default: 1)')
    parser.add_argument('--steps', type=int, default=25, help=
        'Sampling steps (default: 25)')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help=
        'Guidance scale (default: 7.5)')
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--data_dir', default='/dataset/imagenet', type=str,
        help='ImageNet dir')
    parser.add_argument('--n_bits_w', default=8, type=int, help=
        'bitwidth for weight quantization')
    parser.add_argument('--channel_wise', action='store_true', default=True,
        help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=8, type=int, help=
        'bitwidth for activation quantization')
    parser.add_argument('--act_quant', action='store_true', default=False,
        help='apply activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')
    parser.add_argument('--calib_num_samples', default=1024, type=int, help
        ='size of the calibration dataset')
    parser.add_argument('--iters_w', default=100, type=int, help=
        'number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help=
        'weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--keep_cpu', action='store_true', help=
        'keep the calibration data on cpu')
    parser.add_argument('--wwq', action='store_true', default=True, help=
        'weight_quant for input in weight reconstruction')
    parser.add_argument('--waq', action='store_true', default=True, help=
        'act_quant for input in weight reconstruction')
    parser.add_argument('--b_start', default=20, type=int, help=
        'temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help=
        'temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help=
        'in the warmup period no regularization is applied')
    parser.add_argument('--lr', default=4e-05, type=float, help=
        'learning rate for LSQ')
    parser.add_argument('--awq', action='store_true', default=True, help=
        'weight_quant for input in activation reconstruction')
    parser.add_argument('--aaq', action='store_true', default=True, help=
        'act_quant for input in activation reconstruction')
    parser.add_argument('--init_wmode', default='minmax', type=str, choices
        =['minmax', 'mse', 'minmax_scale', 'mse_qdiff'], help=
        'init opt mode for weight')
    parser.add_argument('--init_amode', default='mse', type=str, choices=[
        'minmax', 'mse', 'minmax_scale', 'mse_qdiff'], help=
        'init opt mode for activation')
    parser.add_argument('--order', default='together', type=str, choices=[
        'before', 'after', 'together'], help=
        'order about activation compare to weight')
    parser.add_argument('--prob', default=0.5, type=float)
    parser.add_argument('--input_prob', default=0.5, type=float)
    parser.add_argument('--use_adaround', action='store_true')
    parser.add_argument('--calib_im_mode', default='noise_backward_t', type
        =str, choices=['random', 'raw', 'raw_forward_t', 'noise_backward_t'])
    parser.add_argument('--calib_t_mode', default='normal', type=str,
        choices=['random', '1', '-1', 'mean', 'uniform', 'manual', 'normal',
        'poisson'])
    parser.add_argument('--calib_t_mode_normal_mean', default=0.4, type=
        float, help='for adjusting the weights in the normal distribution')
    parser.add_argument('--calib_t_mode_normal_std', default=0.4, type=
        float, help='for adjusting the weights in the normal distribution')
    parser.add_argument('--out_path', default=
        '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-platcv/tsl/code/quantize/PTQ4DM/results/random8-normalmean04std040_ddim250.npz'
        , type=str)
    args = parser.parse_args()
    paddle.device.set_device('gpu')
    main(args.image, args.motion, args.cali_image, args.cali_motion, args.
        seed, args.steps, args.guidance_scale, args)