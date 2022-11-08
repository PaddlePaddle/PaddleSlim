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
import time
import paddle
from ppdet.core.workspace import load_config
from ppdet.core.workspace import create
from ppdet.metrics import COCOMetric
import onnxruntime as ort

from post_process import PPYOLOEPostProcess


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--reader_config',
        type=str,
        default='configs/picodet_reader.yml',
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--model_path',
        type=str,
        default='onnx_file/picodet_s_416_npu_postprocessed.onnx',
        help="onnx filepath")
    parser.add_argument(
        '--include_post_process',
        type=bool,
        default=False,
        help="Whether include post_process or not.")

    return parser


def eval(val_loader, metric, sess):
    inputs_name = [a.name for a in sess.get_inputs()]
    predict_time = 0.0
    time_min = float("inf")
    time_max = float("-inf")
    sample_nums = len(val_loader)
    for batch_id, data in enumerate(val_loader):
        data_all = {k: np.array(v) for k, v in data.items()}
        data_input = {}
        for k, v in data.items():
            if k in inputs_name:
                data_input[k] = np.array(v)

        start_time = time.time()

        outs = sess.run(None, data_input)

        end_time = time.time()
        timed = end_time - start_time
        time_min = min(time_min, timed)
        time_max = max(time_max, timed)
        predict_time += timed

        res = {}
        if not FLAGS.include_post_process:
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
    time_avg = predict_time / sample_nums
    print("[Benchmark]Inference time(ms): min={}, max={}, avg={}".format(
        round(time_min * 1000, 2),
        round(time_max * 1000, 1), round(time_avg * 1000, 1)))
    print("[Benchmark] COCO mAP: {}".format(map_res["bbox"][0]))
    sys.stdout.flush()


def main():

    reader_cfg = load_config(FLAGS.reader_config)

    dataset = reader_cfg['EvalDataset']
    val_loader = create('EvalReader')(reader_cfg['EvalDataset'],
                                      reader_cfg['worker_num'],
                                      return_list=True)
    clsid2catid = {v: k for k, v in dataset.catid2clsid.items()}
    anno_file = dataset.get_anno()
    metric = COCOMetric(
        anno_file=anno_file, clsid2catid=clsid2catid, IouType='bbox')

    providers = ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.optimized_model_filepath = "./optimize_model.onnx"
    sess = ort.InferenceSession(
        FLAGS.model_path, providers=providers, sess_options=sess_options)
    eval(val_loader, metric, sess)


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()

    # DataLoader need run on cpu
    paddle.set_device("cpu")

    main()
