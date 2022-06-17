# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import numpy as np
import argparse
import paddle
import copy
import cv2
from ppdet.core.workspace import load_config, merge_config
from ppdet.core.workspace import create
from ppdet.metrics import COCOMetric
from paddleslim.auto_compression.config_helpers import load_config as load_slim_config
from paddleslim.auto_compression import AutoCompression
from paddleslim.quant import quant_post_static
from keypoint_utils import HRNetPostProcess, KeyPointTopDownCOCOEval, transform_preds

def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='output',
        help="directory to save compressed model.")
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")
    parser.add_argument(
        '--eval', type=bool, default=False, help="whether to run evaluation.")
    parser.add_argument(
        '--quant', type=bool, default=False, help="whether to run evaluation.")
    return parser


def print_arguments(args):
    print('-----------  Running Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------')


def reader_wrapper(reader, input_list):
    def gen():
        for data in reader:
            in_dict = {}
            for input_name in input_list:
                in_dict[input_name] = data[input_name]
            yield in_dict

    return gen

def flip_back(output_flipped, matched_parts):
    assert output_flipped.ndim == 4,\
            'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped

def eval(config):

    place = paddle.CUDAPlace(0) if FLAGS.devices == 'gpu' else paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    val_program, feed_target_names, fetch_targets = paddle.fluid.io.load_inference_model(
        config["model_dir"],
        exe,
        model_filename=config["model_filename"],
        params_filename=config["params_filename"], )
    dataset.check_or_download_dataset()
    anno_file = dataset.get_anno()
    metric = KeyPointTopDownCOCOEval(anno_file, len(dataset), 17, 'output_eval')
    post_process = HRNetPostProcess()
    for batch_id, data in enumerate(val_loader):
        data_all = {k: np.array(v) for k, v in data.items()}
        data_input = {}
        for k, v in data.items():
            if k in config['input_list']:
                data_input[k] = np.array(v)
        outs = exe.run(val_program,
                       feed=data_input,
                       fetch_list=fetch_targets,
                       return_numpy=False)
        
        data_input['image'] = np.flip(data_input['image'], [3])        
        output_flipped = exe.run(val_program,
                                 feed=data_input,
                                 fetch_list=fetch_targets,
                                 return_numpy=False)
        
        output_flipped = np.array(output_flipped[0])
        flip_perm = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        output_flipped = flip_back(output_flipped, flip_perm)
        output_flipped[:, :, :, 1:] = copy.copy(output_flipped)[:, :, :, 0:-1]
        hrnet_outputs = (np.array(outs[0]) + output_flipped) * 0.5
        imshape = (np.array(data['im_shape'])
                   )[:, ::-1] if 'im_shape' in data else None
        center = np.array(data['center']) if 'center' in data else np.round(imshape / 2.)
        scale = np.array(data['scale']) if 'scale' in data else imshape / 200.
        outputs = post_process(hrnet_outputs, center, scale)
        outputs = {'keypoint': outputs}
        metric.update(data_all, outputs)
        if batch_id % 100 == 0:
            print('Eval iter:', batch_id)
    metric.accumulate()
    metric.log()
    metric.reset()


def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):
    dataset.check_or_download_dataset()
    anno_file = dataset.get_anno()
    metric = KeyPointTopDownCOCOEval(anno_file, len(dataset), 17, 'output_eval')
    post_process = HRNetPostProcess()
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
        
        data_input['image'] = np.flip(data_input['image'], [3])        
        output_flipped = exe.run(compiled_test_program,
                                 feed=data_input,
                                 fetch_list=test_fetch_list,
                                 return_numpy=False)
        
        output_flipped = np.array(output_flipped[0])
        flip_perm = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        output_flipped = flip_back(output_flipped, flip_perm)
        output_flipped[:, :, :, 1:] = copy.copy(output_flipped)[:, :, :, 0:-1]
        hrnet_outputs = (np.array(outs[0]) + output_flipped) * 0.5
        imshape = (np.array(data['im_shape'])
                   )[:, ::-1] if 'im_shape' in data else None
        center = np.array(data['center']) if 'center' in data else np.round(imshape / 2.)
        scale = np.array(data['scale']) if 'scale' in data else imshape / 200.
        outputs = post_process(hrnet_outputs, center, scale)
        outputs = {'keypoint': outputs}

        metric.update(data_all, outputs)
        if batch_id % 100 == 0:
            print('Eval iter:', batch_id)
    metric.accumulate()
    metric.log()
    map_res = metric.get_results()
    metric.reset()
    return map_res['keypoint'][0]


def main():
    compress_config, train_config, global_config = load_slim_config(
        FLAGS.config_path)
    reader_cfg = load_config(global_config['reader_config'])

    train_loader = create('EvalReader')(reader_cfg['TrainDataset'],
                                        reader_cfg['worker_num'],
                                        return_list=True)
    train_loader = reader_wrapper(train_loader, global_config['input_list'])

    global dataset
    dataset = reader_cfg['EvalDataset']
    global val_loader
    val_loader = create('EvalReader')(reader_cfg['EvalDataset'],
                                      reader_cfg['worker_num'],
                                      return_list=True)

    if FLAGS.eval:
        eval(global_config)
        sys.exit(0)

    if 'Evaluation' in global_config.keys() and global_config['Evaluation']:
        eval_func = eval_function
    else:
        eval_func = None

    ac = AutoCompression(
        model_dir=global_config["model_dir"],
        model_filename=global_config["model_filename"],
        params_filename=global_config["params_filename"],
        save_dir=FLAGS.save_dir,
        strategy_config=compress_config,
        train_config=train_config,
        train_dataloader=train_loader,
        eval_callback=eval_func)

    ac.compress()


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)

    assert FLAGS.devices in ['cpu', 'gpu', 'xpu', 'npu']
    paddle.set_device(FLAGS.devices)

    main()
