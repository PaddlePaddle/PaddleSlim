import os
import sys
import paddle
import paddle.nn as nn
import numpy as np
import argparse
import importlib
#import timm
import copy
import time

from utils.calibrator import QuantCalibrator
from utils.block_recon import BlockReconstructor
from utils.mlp_recon import MLPReconstructor
from utils.wrap_net import wrap_modules_in_net, wrap_reparamed_modules_in_net
from datetime import datetime
import logging

import random
import math
from utils.datasets import get_dataloader,get_dataset,get_dataloader_,get_dataset_

from utils.utils import AverageMeter
from utils.utils import all_reduce_mean
from models.interpolate_position_embedding import interpolate_position_embedding
from quant_layers import MinMaxQuantMatMul, MinMaxQuantConv2d, MinMaxQuantLinear

while True:
    try:
        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime("%Y%m%d_%H%M")
        root_path = './checkpoints/quant_result/{}'.format(formatted_timestamp)
        os.makedirs(root_path)
        break
    except FileExistsError:
        time.sleep(10)
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                        logging.FileHandler('{}/output.log'.format(root_path)),
                        logging.StreamHandler()
                    ])


import builtins
original_print = builtins.print
def custom_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    original_print(*args, **kwargs)
builtins.print = custom_print

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default="deit_small",
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large',
                                 'deit_tiny', 'deit_small', 'deit_base', 
                                 'swin_tiny', 'swin_small', 'swin_base', 'swin_base_384'],
                        help="model")
    parser.add_argument('--config', type=str, default="./configs/vit_config.py",
                        help="File path to import Config class from")
    # parser.add_argument('--dataset', default="/dataset/imagenet/",
    #                     help='path to dataset')
    parser.add_argument("--calib-size", default=argparse.SUPPRESS,
                        type=int, help="size of calibration set")
    parser.add_argument("--calib-batch-size", default=argparse.SUPPRESS,
                        type=int, help="batchsize of calibration set")
    parser.add_argument("--val-batch-size", default=200,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--num-workers", default=8, type=int,
                        help="number of data loading workers (default: 8)")
    parser.add_argument("--device", default="cuda", type=str, help="device")

    parser.add_argument('--reconstruct-mlp', action='store_true', help='reconstruct mlp with ReLU function.')
    parser.add_argument('--load-reconstruct-checkpoint', type=str, default=None, help='Path to the reconstructed checkpoint.')
    parser.add_argument('--test-reconstruct-checkpoint', action='store_true', help='validate the reconstructed checkpoint.')
    
    calibrate_mode_group = parser.add_mutually_exclusive_group()
    calibrate_mode_group.add_argument('--calibrate', action='store_true', help="Calibrate the model")
    calibrate_mode_group.add_argument('--load-calibrate-checkpoint', type=str, default=None, help="Path to the calibrated checkpoint.")
    parser.add_argument('--test-calibrate-checkpoint', action='store_true', help='validate the calibrated checkpoint.')

    optimize_mode_group = parser.add_mutually_exclusive_group()
    optimize_mode_group.add_argument('--optimize', action='store_true', help="Optimize the model")
    optimize_mode_group.add_argument('--load-optimize-checkpoint', type=str, default=None, help="Path to the optimized checkpoint.")
    parser.add_argument('--test-optimize-checkpoint', action='store_true', help='validate the optimized checkpoint.')

    parser.add_argument("--print-freq", default=10,
                        type=int, help="print frequency")
    parser.add_argument("--seed", default=3407, type=int, help="seed")
    parser.add_argument('--w_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of weights')
    parser.add_argument('--a_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of activation')
    parser.add_argument("--recon-metric", type=str, default=argparse.SUPPRESS, choices=['hessian_perturb', 'mse', 'mae'], 
                        help='mlp reconstruction metric')
    parser.add_argument("--calib-metric", type=str, default=argparse.SUPPRESS, choices=['mse', 'mae'], 
                        help='calibration metric')
    parser.add_argument("--optim-metric", type=str, default=argparse.SUPPRESS, choices=['hessian', 'hessian_perturb', 'mse', 'mae'], 
                        help='optimization metric')
    parser.add_argument('--optim-mode', type=str, default=argparse.SUPPRESS, choices=['qinp', 'rinp', 'qdrop'], 
                        help='`qinp`: use quanted input; `rinp`: use raw input; `qdrop`: use qdrop input.')
    parser.add_argument('--drop-prob', type=float, default=argparse.SUPPRESS, 
                        help='dropping rate in qdrop. set `drop-prob = 1.0` if do not use qdrop.')
    parser.add_argument('--pct', type=float, default=argparse.SUPPRESS, help='clamp percentile of mlp.fc2.')
    # for model
    parser.add_argument('-cfg', type=str, default=None)
    parser.add_argument('-dataset', type=str, default='imagenet2012')
    parser.add_argument('-data_path', type=str, default='/irip/wangshihe_2023/data/ILSVRC/')
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-batch_size_eval', type=int, default=None)
    parser.add_argument('-image_size', type=int, default=None)
    parser.add_argument('-accum_iter', type=int, default=None)
    parser.add_argument('-pretrained', type=str, default=None)
    parser.add_argument('-resume', type=str, default=None)
    parser.add_argument('-last_epoch', type=int, default=None)
    parser.add_argument('-eval', action='store_true')
    parser.add_argument('-amp', action='store_true')
    parser.add_argument('-output', type=str, default=None)
    parser.add_argument('-teacher_model_path', type=str, default=None)
    
    parser.add_argument('--load-reconstruct-checkpoint-i', type=str, default=None, help='Path to the reconstructed checkpoint.')
    
    
    return parser


def get_cur_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_model(model, args, cfg, mode='calibrate'):
    assert mode in ['calibrate', 'optimize','mlp']
    if mode == 'mlp':
        auto_name = '{}_mlp.pth'.format(
            args.model)
    elif mode == 'calibrate':
        auto_name = '{}_w{}_a{}_calibsize_{}_{}.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.calib_size, cfg.calib_metric)
    else:
        auto_name = '{}_w{}_a{}_optimsize_{}_{}_{}{}.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.optim_size, cfg.optim_metric, cfg.optim_mode, '_recon' if args.reconstruct_mlp else '')
    save_path = os.path.join(root_path, auto_name)

    state_dict = dict()
    state_dict['model'] = model.state_dict()
    logging.info(f"Saving checkpoint to {save_path}")
    paddle.save(state_dict, save_path)


def load_model_(model, args,model_type, mode='calibrate'):
    assert mode in ['calibrate', 'optimize']
    ckpt_path = args.load_calibrate_checkpoint if mode == 'calibrate' else args.load_optimize_checkpoint
    model_state = paddle.load(ckpt_path)
    if 'model' in model_state: # load state_dict with multi items: model, optimier, and epoch
        # pretrain only load model weight, opt and epoch are ignored
        if 'model_ema' in model_state:
            model_state = model_state['model_ema']
        else:
            model_state = model_state['model']
        # if model_type=='swin':
        #     for key in [k for k in model_state.keys() if 'relative_position_index' in k]:
        #         del model_state[key]
        #     # delete relative_coords_table since it is always re-initialized
        #     for key in [k for k in model_state.keys() if 'relative_coords_table' in k]:
        #         del model_state[key]
        #     # delete attn_mask since it is always re-initialized
        #     for key in [k for k in model_state.keys() if 'attn_mask' in k]:
        #         del model_state[key]
    interpolate_position_embedding(model, model_state,model_type)

    for name, module in model.named_sublayers():
        if hasattr(module, 'qmode'):
            module.calibrated = True
            module.qmode = 'quant_forward'
        if isinstance(module, nn.Linear) and 'reduction' in name:
            bias=paddle.zeros(module.weight.shape[1])
            module.bias = paddle.create_parameter(shape=bias.shape,
                    dtype=bias.dtype,
                        default_initializer=paddle.nn.initializer.Assign(bias))
        quantizer_attrs = ['a_quantizer', 'w_quantizer', 'A_quantizer', 'B_quantizer']
        for attr in quantizer_attrs:
            if hasattr(module, attr):
                getattr(module, attr).inited = True
                ckpt_name = name + '.' + attr + '.scale'
                getattr(module, attr).scale.data = model_state[ckpt_name].clone()
 
    result = model.set_state_dict(model_state)
    logging.info(str(result))
    model.eval()
    return model
def load_model(model,model_ch,model_type):
    model_state = paddle.load(model_ch)
    # for name, module in model.named_sublayers(include_self=True):
    #     print(name)
    # for key in model_state.keys():
    #     print(key)
    if 'model' in model_state: # load state_dict with multi items: model, optimier, and epoch
        # pretrain only load model weight, opt and epoch are ignored
        if 'model_ema' in model_state:
            model_state = model_state['model_ema']
        else:
            model_state = model_state['model']
        # if model_type=='swin':
        #     for key in [k for k in model_state.keys() if 'relative_position_index' in k]:
        #         del model_state[key]
        #     # delete relative_coords_table since it is always re-initialized
        #     for key in [k for k in model_state.keys() if 'relative_coords_table' in k]:
        #         del model_state[key]
        #     # delete attn_mask since it is always re-initialized
        #     for key in [k for k in model_state.keys() if 'attn_mask' in k]:
        #         del model_state[key]
    interpolate_position_embedding(model, model_state,model_type)
    
    model.set_state_dict(model_state)
    message = f"----- Pretrained: Load model state from {model_ch}"
    logging.info(message)

@paddle.no_grad()
def validate(dataloader,
             model,
             criterion,
             total_batches,
             debug_steps=10,
             local_logger=None,
             master_logger=None):
    """Validation for the whole dataset
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        total_batches: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        local_logger: logger for local process/gpu, default: None
        master_logger: logger for main process, default: None
    Returns:
        val_loss_meter.avg: float, average loss on current process/gpu
        val_acc1_meter.avg: float, average top1 accuracy on current processes/gpus
        val_acc5_meter.avg: float, average top5 accuracy on current processes/gpus
        master_loss_meter.avg: float, average loss on all processes/gpus
        master_acc1_meter.avg: float, average top1 accuracy on all processes/gpus
        master_acc5_meter.avg: float, average top5 accuracy on all processes/gpus
        val_time: float, validation time
    """
    model.eval()
    val_loss_meter = AverageMeter()
    val_acc1_meter = AverageMeter()
    val_acc5_meter = AverageMeter()
    master_loss_meter = AverageMeter()
    master_acc1_meter = AverageMeter()
    master_acc5_meter = AverageMeter()

    time_st = time.time()

    for batch_id, data in enumerate(dataloader):
        # get data
        images = data[0]
        label = data[1]
        batch_size = images.shape[0]

        output = model(images)
        loss = criterion(output, label)
        loss_value = loss.item()

        pred = paddle.nn.functional.softmax(output)
        acc1 = paddle.metric.accuracy(pred, label.unsqueeze(1)).item()
        acc5 = paddle.metric.accuracy(pred, label.unsqueeze(1), k=5).item()

        # sync from other gpus for overall loss and acc
        master_loss = all_reduce_mean(loss_value)
        master_acc1 = all_reduce_mean(acc1)
        master_acc5 = all_reduce_mean(acc5)
        master_batch_size = all_reduce_mean(batch_size)

        master_loss_meter.update(master_loss, master_batch_size)
        master_acc1_meter.update(master_acc1, master_batch_size)
        master_acc5_meter.update(master_acc5, master_batch_size)
        val_loss_meter.update(loss_value, batch_size)
        val_acc1_meter.update(acc1, batch_size)
        val_acc5_meter.update(acc5, batch_size)

        if batch_id % debug_steps == 0 or batch_id==total_batches-1:
            local_message = (f"Step[{batch_id:04d}/{total_batches:04d}], "
                                f"Avg Loss: {val_loss_meter.avg:.4f}, "
                                f"Avg Acc@1: {val_acc1_meter.avg:.4f}, "
                                f"Avg Acc@5: {val_acc5_meter.avg:.4f}")
            master_message = (f"Step[{batch_id:04d}/{total_batches:04d}], "
                                f"Avg Loss: {master_loss_meter.avg:.4f}, "
                                f"Avg Acc@1: {master_acc1_meter.avg:.4f}, "
                                f"Avg Acc@5: {master_acc5_meter.avg:.4f}")
            #write_log(local_logger, master_logger, local_message, master_message)
            logging.info(local_message+master_message)
    #paddle.distributed.barrier()
    val_time = time.time() - time_st
    return (val_loss_meter.avg,
            val_acc1_meter.avg,
            val_acc5_meter.avg,
            master_loss_meter.avg,
            master_acc1_meter.avg,
            master_acc5_meter.avg,
            val_time)

def main(args):
    
    #paddle.utils.run_check()
    logging.info("{} - start the process.".format(get_cur_time()))
    logging.info(str(args))
    dir_path = os.path.dirname(os.path.abspath(args.config))
    if dir_path not in sys.path:
        sys.path.append(dir_path)
    module_name = os.path.splitext(os.path.basename(args.config))[0]
    imported_module = importlib.import_module(module_name)
    Config = getattr(imported_module, 'Config')
    logging.info("Successfully imported Config class!")
        
    cfg = Config()
    cfg.calib_size = args.calib_size if hasattr(args, 'calib_size') else cfg.calib_size
    cfg.calib_batch_size = args.calib_batch_size if hasattr(args, 'calib_batch_size') else cfg.calib_batch_size
    cfg.recon_metric = args.recon_metric if hasattr(args, 'recon_metric') else cfg.recon_metric
    cfg.calib_metric = args.calib_metric if hasattr(args, 'calib_metric') else cfg.calib_metric
    cfg.optim_metric = args.optim_metric if hasattr(args, 'optim_metric') else cfg.optim_metric
    cfg.optim_mode = args.optim_mode if hasattr(args, 'optim_mode') else cfg.optim_mode
    cfg.drop_prob = args.drop_prob if hasattr(args, 'drop_prob') else cfg.drop_prob
    cfg.reconstruct_mlp = args.reconstruct_mlp
    cfg.pct = args.pct if hasattr(args, 'pct') else cfg.pct
    cfg.w_bit = args.w_bit if hasattr(args, 'w_bit') else cfg.w_bit
    cfg.a_bit = args.a_bit if hasattr(args, 'a_bit') else cfg.a_bit
    for name, value in vars(cfg).items():
        logging.info(f"{name}: {value}")

    model_type=args.model.split('_')[0]
    logging.info("model_type......{}".format(model_type))
    if model_type=='swin': 
        from models.configs_swin.config import update_config,get_config
        from models.swin import build_swin as build_model
    elif model_type=='deit': 
        from models.configs_deit.config import update_config,get_config
        from models.deit import build_deit as build_model
    else:
        from models.configs_vit.config import update_config,get_config
        from models.vit import build_vit as build_model

    config = update_config(get_config(), args)
    
    
    paddle.device.set_device('gpu')
    paddle.distributed.init_parallel_env()
    world_size = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    seed = args.seed + local_rank
    logging.info('seed:'.format(seed))
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    logging.info('Building model ...')
    model = build_model(config)
    criterion = paddle.nn.CrossEntropyLoss()
    
    
    logging.info('Building validation dataloader ...')
    dataset_train = get_dataset(config, is_train=True,model_type=model_type) 
    dataset_val = get_dataset(config, is_train=False)
    dataloader_val = get_dataloader(config, dataset_val, False, True)
    total_batch_val = len(dataloader_val)
    
    

    assert os.path.isfile(config.MODEL.PRETRAINED) is True
    load_model(model,config.MODEL.PRETRAINED,model_type)
    

    #model = paddle.DataParallel(model)
    model.eval()
    full_model = copy.deepcopy(model)
    full_model.eval()

    def datal(size,batch_size):
        np.random.seed(seed)
        inds = np.random.permutation(len(dataset_train))[:size]
        img_path_list=[dataset_train.img_path_list[i] for i in inds]
        label_list=[dataset_train.label_list[i] for i in inds]
        optim_data = get_dataset_(config,img_path_list,label_list, is_train=True,model_type=model_type)
        return get_dataloader_(config, optim_data,batch_size, False, True)
    if args.reconstruct_mlp:
        for name, module in model.named_sublayers(include_self=True):
            #print(name)
            if name.split('.')[-1] == 'mlp':
                module.act = nn.ReLU()
        if args.load_reconstruct_checkpoint is not None:
            load_model(model,args.load_reconstruct_checkpoint,model_type)
            full_model = copy.deepcopy(model)
            model.eval()
            if args.test_reconstruct_checkpoint:
                val_loss, val_acc1, val_acc5, avg_loss, avg_acc1, avg_acc5, val_time = validate(
                        dataloader=dataloader_val,
                        model=model,
                        criterion=criterion,
                        total_batches=total_batch_val,
                        debug_steps=50)
        elif args.load_calibrate_checkpoint is None:
            if args.load_reconstruct_checkpoint_i is not None:
                load_model(model,args.load_reconstruct_checkpoint_i,model_type)
                # val_loss, val_acc1, val_acc5, avg_loss, avg_acc1, avg_acc5, val_time = validate(
                #         dataloader=dataloader_val,
                #         model=model,
                #         criterion=criterion,
                #         total_batches=total_batch_val,
                #         debug_steps=50)
            logging.info('Building calibrator ...')
            dataloader_optim = datal(cfg.optim_size,cfg.optim_batch_size)
            logging.info('{} - Start reconstructing MLP blocks ...'.format(get_cur_time()))
            mlp_reconstructor = MLPReconstructor(model, full_model, dataloader_optim, metric=cfg.recon_metric, temp=cfg.temp)
            mlp_reconstructor.reconstruct_model(pct=cfg.pct,model_name=args.model,root_path=root_path)
            logging.info("{} - MLP reconstruction finished.".format(get_cur_time()))

            save_path = os.path.join(root_path, '{}_reconstructed.pth'.format(args.model))
            state_dict = dict()
            state_dict['model'] = model.state_dict()
            paddle.save(state_dict, save_path)
            logging.info ("----- Save model: {}".format(save_path))
            
            logging.info('Validating after model reconstruction ...')
            val_loss, val_acc1, val_acc5, avg_loss, avg_acc1, avg_acc5, val_time = validate(
                        dataloader=dataloader_val,
                        model=model,
                        criterion=criterion,
                        total_batches=total_batch_val,
                        debug_steps=50)

    reparam = args.load_calibrate_checkpoint is None and args.load_optimize_checkpoint is None
    logging.info('Wraping quantiztion modules (reparam: {}, recon: {}) ...'.format(reparam, args.reconstruct_mlp))
    model = wrap_modules_in_net(model, cfg, reparam=reparam, recon=args.reconstruct_mlp)
    model.eval()
   
    if not args.load_optimize_checkpoint:
        if args.load_calibrate_checkpoint:
            load_model_(model,args,model_type,mode='calibrate')
            logging.info(f"Restoring checkpoint from '{args.load_calibrate_checkpoint}'")
            if args.test_calibrate_checkpoint:
                for name, module in model.named_sublayers(include_self=True):
                    if hasattr(module, 'qmode'):
                        module.qmode = 'quant_forward'
                val_loss, val_acc1, val_acc5, avg_loss, avg_acc1, avg_acc5, val_time = validate(
                        dataloader=dataloader_val,
                        model=model,
                        criterion=criterion,
                        total_batches=total_batch_val,
                        debug_steps=50)
                for name, module in model.named_sublayers(include_self=True):
                    if hasattr(module, 'qmode'):
                        module.qmode = 'raw'
            
        else:
            logging.info("{} - start {} guided calibration".format(get_cur_time(), cfg.calib_metric))
            calib_loader = datal(cfg.calib_size, cfg.calib_batch_size)
            quant_calibrator = QuantCalibrator(model, calib_loader)
            quant_calibrator.batching_quant_calib()
            model = wrap_reparamed_modules_in_net(model)
            logging.info("{} - {} guided calibration finished.".format(get_cur_time(), cfg.calib_metric))
            save_model(model, args, cfg, mode='calibrate')
            logging.info('Validating after calibration ...')
            val_loss, val_acc1, val_acc5, avg_loss, avg_acc1, avg_acc5, val_time = validate(
                        dataloader=dataloader_val,
                        model=model,
                        criterion=criterion,
                        total_batches=total_batch_val,
                        debug_steps=50)
    
    calib_loader = datal(cfg.optim_size,cfg.optim_batch_size)
    if args.optimize:
        logging.info('Building calibrator ...')
        logging.info("{} - start {} guided block reconstruction".format(get_cur_time(), cfg.optim_metric))
        block_reconstructor = BlockReconstructor(model, full_model, calib_loader, metric=cfg.optim_metric, temp=cfg.temp, use_mean_hessian=cfg.use_mean_hessian)
        block_reconstructor.reconstruct_model(quant_act=True, mode=cfg.optim_mode, drop_prob=cfg.drop_prob, keep_gpu=cfg.keep_gpu,root_path=root_path)
        logging.info("{} - {} guided block reconstruction finished.".format(get_cur_time(), cfg.optim_metric))
        save_model(model, args, cfg, mode='optimize')
    if args.load_optimize_checkpoint:
        logging.info('Building calibrator ...')
        model = load_model_(model,args,model_type,mode='optimize')
    if args.optimize or args.test_optimize_checkpoint:
        logging.info('Validating on calibration set after block reconstruction ...')
        val_loss, val_acc1, val_acc5, avg_loss, avg_acc1, avg_acc5, val_time = validate(
                        dataloader=calib_loader,
                        model=model,
                        criterion=criterion,
                        total_batches=total_batch_val,
                        debug_steps=50)
        logging.info('Validating on test set after block reconstruction ...')
        val_loss, val_acc1, val_acc5, avg_loss, avg_acc1, avg_acc5, val_time = validate(
                        dataloader=dataloader_val,
                        model=model,
                        criterion=criterion,
                        total_batches=total_batch_val,
                        debug_steps=50)
    logging.info("{} - finished the process.".format(get_cur_time()))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    