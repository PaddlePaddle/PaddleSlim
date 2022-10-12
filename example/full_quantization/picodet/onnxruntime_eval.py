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
from ppdet.core.workspace import load_config
from ppdet.core.workspace import create
from ppdet.metrics import COCOMetric
import onnxruntime as ort

from post_process import PicoDetPostProcess


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

    for batch_id, data in enumerate(val_loader):
        data_all = {k: np.array(v) for k, v in data.items()}
        batch_size = data_all['image'].shape[0]
        data_input = {}
        for k, v in data.items():
            if k in inputs_name:
                data_input[k] = np.array(v)

        outs = sess.run(None, data_input)

        if not FLAGS.include_post_process:
            np_score_list, np_boxes_list = [], []
            for i, out in enumerate(outs):
                np_out = np.array(out)
                if i < 4:
                    num_classes = np_out.shape[-1]
                    np_score_list.append(
                        np_out.reshape(batch_size, -1, num_classes))
                else:
                    box_reg_shape = np_out.shape[-1]
                    np_boxes_list.append(
                        np_out.reshape(batch_size, -1, box_reg_shape))
            post_processor = PicoDetPostProcess(
                data_all['image'].shape[2:],
                data_all['im_shape'],
                data_all['scale_factor'],
                score_threshold=0.01,
                nms_threshold=0.6)
            res = post_processor(np_score_list, np_boxes_list)
        else:
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
