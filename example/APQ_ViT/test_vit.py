import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')
# from test_vit import *
import paddle
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
from itertools import product
import types
from utils.quant_calib import HessianQuantCalibrator, QuantCalibrator
from utils.models import get_net
import time
# from importlib import reload, import_module
import os
import utils.datasets as datasets
import utils.net_wrap as net_wrap
from utils.datasets_image import build_imagenet_data, get_dataset, get_dataloader
from utils.basic import Count
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpu', type=int, default=6)
    parser.add_argument('--multiprocess', action='store_true')
    parser.add_argument('--model_config_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()
    return args


def test_classification(net, test_loader, max_iteration=None, description=None):
    pos = 0
    tot = 0
    i = 0
    max_iteration = len(test_loader) if max_iteration is None else max_iteration
    with paddle.no_grad():
        q = tqdm(test_loader, desc=description)
        for inp, target in q:
            i += 1
            # inp = paddle.to_tensor(inp)
            # target = paddle.to_tensor(target)
            inp = inp.cuda(blocking=True)
            target = target.cuda(blocking=True)
            out = net(inp)
            pos_num = paddle.sum(x=out.argmax(axis=1) == target).item()
            pos += pos_num
            tot += inp.shape[0]
            q.set_postfix({'acc': pos / tot})
            if i >= max_iteration:
                break
    print(pos / tot)
    return pos / tot


# def init_config(config_name):
#     """initialize the config. Use reload to make sure it's fresh one!"""
#     _, _, files = next(os.walk('./configs'))
#     if config_name + '.py' in files:
#         quant_cfg = import_module(f'configs.{config_name}')
#     else:
#         raise NotImplementedError(f'Invalid config name {config_name}')
#     reload(quant_cfg)
#     return quant_cfg

# def process(pid, experiment_process, args_queue, n_gpu):
#     """
#     worker process. 
#     """
#     gpu_id = pid % n_gpu
#     os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_id}'
#     tot_run = 0
#     while args_queue.qsize():
#         test_args = args_queue.get()
#         print(f'Run {test_args} on pid={pid} gpu_id={gpu_id}')
#         experiment_process(**test_args)
#         time.sleep(0.5)
#         tot_run += 1
#     print(f'{pid} tot_run {tot_run}')


class PTQConfig:
    def __init__(self, bit=8):
        self.bit = bit
        self.no_softmax = False
        self.no_postgelu = False
        conv_fc_name_list = ['qconv', 'qlinear_qkv', 'qlinear_proj', 'qlinear_MLP_1', 'qlinear_MLP_2', 'qlinear_classifier', 'qlinear_reduction']
        matmul_name_list = ['qmatmul_qk', 'qmatmul_scorev']
        self.w_bit = {name: self.bit for name in conv_fc_name_list}
        self.a_bit = {name: self.bit for name in conv_fc_name_list}
        self.A_bit = {name: self.bit for name in matmul_name_list}
        self.B_bit = {name: self.bit for name in matmul_name_list}
        self.ptqsl_conv2d_kwargs = {'metric': 'hessian', 'eq_alpha': 0.9, 'eq_beta': 
            1.2, 'eq_n': 100, 'search_round': 3, 'n_V': 1, 'n_H': 1}
        self.ptqsl_linear_kwargs = {'metric': 'apq_hessian', 'eq_alpha': 0.7, 'eq_beta': 
            1.1, 'eq_n': 100, 'search_round': 3, 'n_V': 1, 'n_H': 1, 'n_a': 1,
            'bias_correction': True}
        self.ptqsl_matmul_kwargs = {'metric': 'apq_hessian', 'eq_alpha': 0.9, 'eq_beta': 
            1.2, 'eq_n': 100, 'search_round': 3, 'n_G_A': 1, 'n_V_A': 1, 'n_H_A': 1,
            'n_G_B': 1, 'n_V_B': 1, 'n_H_B': 1}
        

def test_all(name, checkpoint_path=None, calib_size=32, config_name='PTQ4ViT', config_path=None, model_path=None):
    # quant_cfg = init_config(config_name)
    # quant_cfg = cfg_modifier(quant_cfg)
    quant_cfg = PTQConfig()
    model_config, net = get_net(name, config_path=config_path, model_path=model_path)

    # train_loader, test_loader, calib_loader = build_imagenet_data('/mnt/disk1/datasets/imagenet')
    dataset_train = get_dataset(model_config, is_train=True) if not model_config.EVAL else None
    dataset_val = get_dataset(model_config, is_train=False)
    calib_loader = get_dataloader(model_config, dataset_val, True, True, get_calib_dataloader=True, calib_size=calib_size)
    test_loader = get_dataloader(model_config, dataset_val, False, True)

    acc_fp = test_classification(net, test_loader, description=quant_cfg.
        ptqsl_linear_kwargs['metric'])
    print(f'FP accuracy: {acc_fp} \n\n')
    # import pdb; pdb.set_trace()
    # cnt = Count(net)
    wrapped_modules = net_wrap.wrap_modules_in_net(net, quant_cfg)

    calib_start_time = time.time()
  
    quant_calibrator = HessianQuantCalibrator(net, wrapped_modules,
        calib_loader, sequential=False, batch_size=4)
    quant_calibrator.batching_quant_calib()
    
    acc = test_classification(net, test_loader, description=quant_cfg.
        ptqsl_linear_kwargs['metric'])
    calib_end_time = time.time()
    print(f'model: {name} \n')
    print(f'calibration size: {calib_size} \n')
    print(f'bit settings: {quant_cfg.bit} \n')
    print(f'config: {config_name} \n')
    print(f'ptqsl_conv2d_kwargs: {quant_cfg.ptqsl_conv2d_kwargs} \n')
    print(f'ptqsl_linear_kwargs: {quant_cfg.ptqsl_linear_kwargs} \n')
    print(f'ptqsl_matmul_kwargs: {quant_cfg.ptqsl_matmul_kwargs} \n')
    print(f'calibration time: {(calib_end_time - calib_start_time) / 60}min \n'
        )
    print(f'accuracy: {acc} \n\n')


if __name__ == '__main__':
    args = parse_args()
    
    test_all(name='vit_base_patch16_224', \
        config_name='PTQ4ViT', \
        config_path=args.model_config_path, \
        model_path=args.model_path)
        