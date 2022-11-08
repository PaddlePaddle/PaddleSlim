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
from tqdm import tqdm
import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.core.workspace import create
from ppdet.metrics import COCOMetric
from paddleslim.common import load_config as load_slim_config
from paddleslim.common.dataloader import get_feed_vars
from paddleslim.quant.analysis import AnalysisQuant

from post_process import PPYOLOEPostProcess


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of analysis config.",
        required=True)
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")
    return parser


def reader_wrapper(reader, input_list):
    def gen():
        for data in reader:
            in_dict = {}
            if isinstance(input_list, list):
                for input_name in input_list:
                    in_dict[input_name] = data[input_name]
            yield in_dict

    return gen


def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):
    metric = global_config['metric']
    for batch_id, data in enumerate(val_loader):
        data_all = {k: np.array(v) for k, v in data.items()}
        data_input = {}
        for k, v in data.items():
            if isinstance(global_config['input_list'], list):
                if k in test_feed_names:
                    data_input[k] = np.array(v)
            elif isinstance(global_config['input_list'], dict):
                if k in global_config['input_list'].keys():
                    data_input[global_config['input_list'][k]] = np.array(v)
        outs = exe.run(compiled_test_program,
                       feed=data_input,
                       fetch_list=test_fetch_list,
                       return_numpy=False)
        res = {}
        if 'exclude_nms' in global_config and global_config['exclude_nms']:
            postprocess = PPYOLOEPostProcess(
                score_threshold=0.01, nms_threshold=0.6)
            res = postprocess(np.array(outs[0]), data_all['scale_factor'])
        else:
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


def main():

    global global_config
    all_config = load_slim_config(FLAGS.config_path)
    assert "Global" in all_config, f"Key 'Global' not found in config file. \n{all_config}"
    global_config = all_config["Global"]
    ptq_config = all_config['PTQ']
    global_config['input_list'] = get_feed_vars(
        global_config['model_dir'], global_config['model_filename'],
        global_config['params_filename'])
    reader_cfg = load_config(global_config['reader_config'])

    train_loader = create('EvalReader')(reader_cfg['TrainDataset'],
                                        reader_cfg['worker_num'],
                                        return_list=True)
    train_loader = reader_wrapper(train_loader, global_config['input_list'])

    dataset = reader_cfg['EvalDataset']
    global val_loader
    _eval_batch_sampler = paddle.io.BatchSampler(
        dataset, batch_size=reader_cfg['EvalReader']['batch_size'])
    val_loader = create('EvalReader')(dataset,
                                      reader_cfg['worker_num'],
                                      batch_sampler=_eval_batch_sampler,
                                      return_list=True)
    global num_classes
    num_classes = reader_cfg['num_classes']
    clsid2catid = {v: k for k, v in dataset.catid2clsid.items()}
    anno_file = dataset.get_anno()
    metric = COCOMetric(
        anno_file=anno_file, clsid2catid=clsid2catid, IouType='bbox')
    global_config['metric'] = metric

    analyzer = AnalysisQuant(
        model_dir=global_config["model_dir"],
        model_filename=global_config["model_filename"],
        params_filename=global_config["params_filename"],
        eval_function=eval_function,
        data_loader=train_loader,
        resume=True,
        save_dir='output',
        ptq_config=ptq_config)

    # plot the boxplot of activations of quantizable weights
    analyzer.plot_activation_distribution()

    # get the rank of sensitivity of each quantized layer
    # plot the histogram plot of best and worst activations and weights if plot_hist is True
    analyzer.compute_quant_sensitivity(plot_hist=True)

    # get the quantized model that satisfies target metric you set
    analyzer.get_target_quant_model(target_metric=0.25)


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()

    assert FLAGS.devices in ['cpu', 'gpu', 'xpu', 'npu']
    paddle.set_device(FLAGS.devices)

    main()
