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
add_arg('save_dir',                    str,    'output',         "directory to save compressed model.")
add_arg('devices',                     str,    'gpu',        "which device used to compress.")
add_arg('batch_size',                  int,    1,            "train batch size.")
add_arg('config_path',                 str,    None,         "path of compression strategy config.")
add_arg('eval',                  bool,    False,            "whether to run evaluation.")
# yapf: enable


def reader_wrapper(reader, input_list):
    def gen():
        for data in reader:
            in_dict = {}
            for input_name in input_list:
                in_dict[input_name] = data[input_name]
            yield in_dict

    return gen


def eval(args, compress_config):

    place = paddle.CUDAPlace(0) if args.devices == 'gpu' else paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    val_program, feed_target_names, fetch_targets = paddle.fluid.io.load_inference_model(
        compress_config["model_dir"],
        exe,
        model_filename=compress_config["model_filename"],
        params_filename=compress_config["params_filename"], )
    clsid2catid = {v: k for k, v in dataset.catid2clsid.items()}

    anno_file = dataset.get_anno()
    metric = COCOMetric(
        anno_file=anno_file, clsid2catid=clsid2catid, bias=0, IouType='bbox')
    for batch_id, data in enumerate(val_loader):
        data_all = {k: np.array(v) for k, v in data.items()}
        data_input = {}
        for k, v in data.items():
            if k in compress_config['input_list']:
                data_input[k] = np.array(v)
        outs = exe.run(val_program,
                       feed=data_input,
                       fetch_list=fetch_targets,
                       return_numpy=False)
        res = {}
        for out in outs:
            v = np.array(out)
            if len(v.shape) > 1:
                res['bbox'] = v
            else:
                res['bbox_num'] = v

        metric.update(data_all, res)
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
        data_all = {k: np.array(v) for k, v in data.items()}
        data_input = {}
        for k, v in data.items():
            if k in test_feed_names:
                data_input[k] = np.array(v)
        outs = exe.run(compiled_test_program,
                       feed=data_input,
                       fetch_list=test_fetch_list,
                       return_numpy=False)
        res = {}
        for out in outs:
            v = np.array(out)
            if len(v.shape) > 1:
                res['bbox'] = v
            else:
                res['bbox_num'] = v

        metric.update(data_all, res)
        if batch_id % 100 == 0:
            print('Eval iter:', batch_id)
    metric.accumulate()
    metric.log()
    map_res = metric.get_results()
    metric.reset()
    return map_res['bbox'][0]


def main(args):
    compress_config, train_config = load_slim_config(args.config_path)
    reader_cfg = load_config(compress_config['reader_config'])

    train_loader = create('EvalReader')(reader_cfg['TrainDataset'],
                                        reader_cfg['worker_num'],
                                        return_list=True)
    train_loader = reader_wrapper(train_loader, compress_config['input_list'])

    global dataset
    dataset = reader_cfg['EvalDataset']
    global val_loader
    val_loader = create('EvalReader')(reader_cfg['EvalDataset'],
                                      reader_cfg['worker_num'],
                                      return_list=True)

    if args.eval:
        eval(args, compress_config)
        sys.exit(0)

    if 'Evaluation' in compress_config.keys() and compress_config['Evaluation']:
        eval_func = eval_function
    else:
        eval_func = None

    ac = AutoCompression(
        model_dir=compress_config["model_dir"],
        model_filename=compress_config["model_filename"],
        params_filename=compress_config["params_filename"],
        save_dir=args.save_dir,
        strategy_config=compress_config,
        train_config=train_config,
        train_dataloader=train_loader,
        eval_callback=eval_func,
        devices=args.devices)

    ac.compress()


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    paddle.enable_static()
    main(args)
