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
from ppdet.metrics import COCOMetric, VOCMetric, KeyPointTopDownCOCOEval
from keypoint_utils import keypoint_post_process
from post_process import PPYOLOEPostProcess
from paddleslim.quant.analysis import AnalysisQuant


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
            elif isinstance(input_list, dict):
                for input_name in input_list.keys():
                    in_dict[input_list[input_name]] = data[input_name]
            yield in_dict

    return gen


def convert_numpy_data(data, metric):
    data_all = {}
    data_all = {k: np.array(v) for k, v in data.items()}
    if isinstance(metric, VOCMetric):
        for k, v in data_all.items():
            if not isinstance(v[0], np.ndarray):
                tmp_list = []
                for t in v:
                    tmp_list.append(np.array(t))
                data_all[k] = np.array(tmp_list)
    else:
        data_all = {k: np.array(v) for k, v in data.items()}
    return data_all


def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):
    with tqdm(
            total=len(val_loader),
            bar_format='Evaluation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
            ncols=80) as t:
        for batch_id, data in enumerate(val_loader):
            data_all = convert_numpy_data(data, metric)
            data_input = {}
            for k, v in data.items():
                if isinstance(config['input_list'], list):
                    if k in test_feed_names:
                        data_input[k] = np.array(v)
                elif isinstance(config['input_list'], dict):
                    if k in config['input_list'].keys():
                        data_input[config['input_list'][k]] = np.array(v)
            outs = exe.run(compiled_test_program,
                           feed=data_input,
                           fetch_list=test_fetch_list,
                           return_numpy=False)
            res = {}
            if 'arch' in config and config['arch'] == 'keypoint':
                res = keypoint_post_process(data, data_input, exe,
                                            compiled_test_program,
                                            test_fetch_list, outs)
            if 'arch' in config and config['arch'] == 'PPYOLOE':
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
            t.update()

    metric.accumulate()
    metric.log()
    map_res = metric.get_results()
    metric.reset()
    map_key = 'keypoint' if 'arch' in config and config[
        'arch'] == 'keypoint' else 'bbox'
    return map_res[map_key][0]


def main():

    global config
    config = load_config(FLAGS.config_path)
    ptq_config = config['PTQ']

    # val dataset is sufficient for PTQ
    data_loader = create('EvalReader')(config['EvalDataset'],
                                       config['worker_num'],
                                       return_list=True)
    ptq_data_loader = reader_wrapper(data_loader, config['input_list'])

    # fast_val_anno_path, such as annotation path of several pictures can accelerate analysis
    dataset = config[
        'FastEvalDataset'] if 'FastEvalDataset' in config else config[
            'EvalDataset']
    global val_loader
    _eval_batch_sampler = paddle.io.BatchSampler(
        dataset, batch_size=config['EvalReader']['batch_size'])
    val_loader = create('EvalReader')(dataset,
                                      config['worker_num'],
                                      batch_sampler=_eval_batch_sampler,
                                      return_list=True)
    global metric
    if config['metric'] == 'COCO':
        clsid2catid = {v: k for k, v in dataset.catid2clsid.items()}
        anno_file = dataset.get_anno()
        metric = COCOMetric(
            anno_file=anno_file, clsid2catid=clsid2catid, IouType='bbox')
    elif config['metric'] == 'VOC':
        metric = VOCMetric(
            label_list=dataset.get_label_list(),
            class_num=config['num_classes'],
            map_type=config['map_type'])
    elif config['metric'] == 'KeyPointTopDownCOCOEval':
        anno_file = dataset.get_anno()
        metric = KeyPointTopDownCOCOEval(anno_file,
                                         len(dataset), 17, 'output_eval')
    else:
        raise ValueError("metric currently only supports COCO and VOC.")

    analyzer = AnalysisQuant(
        model_dir=config["model_dir"],
        model_filename=config["model_filename"],
        params_filename=config["params_filename"],
        eval_function=eval_function,
        data_loader=ptq_data_loader,
        save_dir=config['save_dir'],
        ptq_config=ptq_config,
        resume=True, )

    analyzer.statistical_analyse()
    analyzer.metric_error_analyse()

    if config['get_target_quant_model']:
        if 'FastEvalDataset' in config:
            # change fast_val_loader to full val_loader
            val_loader = data_loader
        # get the quantized model that satisfies target metric you set
        analyzer.get_target_quant_model(target_metric=config['target_metric'])


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()

    assert FLAGS.devices in ['cpu', 'gpu', 'xpu', 'npu']
    paddle.set_device(FLAGS.devices)

    main()
