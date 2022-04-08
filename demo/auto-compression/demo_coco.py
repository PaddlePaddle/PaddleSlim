import os
import sys
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)
import argparse
import functools
from functools import partial
import numpy as np
import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.core.workspace import create
from ppdet.metrics import COCOMetric
from paddleslim.auto_compression.config_helpers import load_config as load_slim_config
from paddleslim.auto_compression import AutoCompression

paddle.enable_static()

from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('model_dir',                   str,    None,         "inference model directory.")
add_arg('model_filename',              str,    None,         "inference model filename.")
add_arg('params_filename',             str,    None,         "inference params filename.")
add_arg('save_dir',                    str,    'output',         "directory to save compressed model.")
add_arg('devices',                     str,    'gpu',        "which device used to compress.")
add_arg('batch_size',                  int,    1,            "train batch size.")
add_arg('config_path',                 str,    None,         "path of compression strategy config.")
add_arg('eval',                  bool,    False,            "whether to run evaluation.")
# yapf: enable


def reader_wrapper(reader):
    def gen():
        for data in reader:
            yield {
                "image": data['image'],
                'im_shape': data['im_shape'],
                'scale_factor': data['scale_factor']
            }

    return gen


def eval():
    dataset = reader_cfg['EvalDataset']
    val_loader = create('TestReader')(dataset,
                                      reader_cfg['worker_num'],
                                      return_list=True)

    place = paddle.CUDAPlace(0) if args.devices == 'gpu' else paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    val_program, feed_target_names, fetch_targets = paddle.fluid.io.load_inference_model(
        args.model_dir,
        exe,
        model_filename=args.model_filename,
        params_filename=args.params_filename)
    clsid2catid = {v: k for k, v in dataset.catid2clsid.items()}

    anno_file = dataset.get_anno()
    metric = COCOMetric(
        anno_file=anno_file, clsid2catid=clsid2catid, bias=0, IouType='bbox')
    for batch_id, data in enumerate(val_loader):
        data_new = {k: np.array(v) for k, v in data.items()}
        outs = exe.run(val_program,
                       feed={
                           'image': data['image'],
                           'im_shape': data['im_shape'],
                           'scale_factor': data['scale_factor']
                       },
                       fetch_list=fetch_targets,
                       return_numpy=False)
        res = {}
        for out in outs:
            v = np.array(out)
            if len(v.shape) > 1:
                res['bbox'] = v
            else:
                res['bbox_num'] = v

        metric.update(data_new, res)
        if batch_id % 100 == 0:
            print('Eval iter:', batch_id)
    metric.accumulate()
    metric.log()
    metric.reset()


def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):
    clsid2catid = {v: k for k, v in dataset.catid2clsid.items()}

    anno_file = dataset.get_anno()
    metric = COCOMetric(
        anno_file=anno_file, clsid2catid=clsid2catid, bias=1, IouType='bbox')
    for batch_id, data in enumerate(val_loader):
        data_new = {k: np.array(v) for k, v in data.items()}
        outs = exe.run(compiled_test_program,
                       feed={
                           'image': data['image'],
                           'im_shape': data['im_shape'],
                           'scale_factor': data['scale_factor']
                       },
                       fetch_list=test_fetch_list,
                       return_numpy=False)
        res = {}
        for out in outs:
            v = np.array(out)
            if len(v.shape) > 1:
                res['bbox'] = v
            else:
                res['bbox_num'] = v

        metric.update(data_new, res)
        if batch_id % 100 == 0:
            print('Eval iter:', batch_id)
    metric.accumulate()
    metric.log()
    map_res = metric.get_results()
    metric.reset()
    return map_res['bbox'][0]


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    paddle.enable_static()
    reader_cfg = load_config('./configs/PaddleDet/yolo_reader.yml')
    if args.eval:
        eval()
        sys.exit(0)

    compress_config, train_config = load_slim_config(args.config_path)

    train_loader = create('TestReader')(reader_cfg['TrainDataset'],
                                        reader_cfg['worker_num'],
                                        return_list=True)
    dataset = reader_cfg['EvalDataset']
    val_loader = create('TestReader')(reader_cfg['EvalDataset'],
                                      reader_cfg['worker_num'],
                                      return_list=True)

    train_dataloader = reader_wrapper(train_loader)

    ac = AutoCompression(
        model_dir=args.model_dir,
        model_filename=args.model_filename,
        params_filename=args.params_filename,
        save_dir=args.save_dir,
        strategy_config=compress_config,
        train_config=train_config,
        train_dataloader=train_dataloader,
        eval_callback=eval_function,
        devices=args.devices)

    ac.compress()
