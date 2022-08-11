#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

import time
import logging
import os
import shutil
import sys

import paddle
from x2paddle.decoder.onnx_decoder import ONNXDecoder
from x2paddle.op_mapper.onnx2paddle.onnx_op_mapper import ONNXOpMapper
from x2paddle.optimizer.optimizer import GraphOptimizer
from x2paddle.utils import ConverterCheck

from . import get_logger
_logger = get_logger(__name__, level=logging.INFO)

__all__ = ['load_onnx_model']


def load_onnx_model(model_path, disable_feedback=False):
    assert model_path.endswith(
        '.onnx'
    ), '{} does not end with .onnx suffix and cannot be loaded.'.format(
        model_path)
    inference_model_path = model_path.rstrip().rstrip('.onnx') + '_infer'
    exe = paddle.static.Executor(paddle.CPUPlace())
    if os.path.exists(os.path.join(
            inference_model_path, 'model.pdmodel')) and os.path.exists(
                os.path.join(inference_model_path, 'model.pdiparams')):
        val_program, feed_target_names, fetch_targets = paddle.static.load_inference_model(
            os.path.join(inference_model_path, 'model'), exe)
        _logger.info('Loaded model from: {}'.format(inference_model_path))
        return val_program, feed_target_names, fetch_targets
    else:
        # onnx to paddle inference model.
        time_info = int(time.time())
        if not disable_feedback:
            ConverterCheck(
                task="ONNX", time_info=time_info, convert_state="Start").start()
        # check onnx installation and version
        try:
            import onnx
            version = onnx.version.version
            v0, v1, v2 = version.split('.')
            version_sum = int(v0) * 100 + int(v1) * 10 + int(v2)
            if version_sum < 160:
                _logger.info("[ERROR] onnx>=1.6.0 is required")
                sys.exit(1)
        except:
            _logger.info(
                "[ERROR] onnx is not installed, use \"pip install onnx==1.6.0\"."
            )
            sys.exit(1)

        try:
            _logger.info("Now translating model from onnx to paddle.")
            model = ONNXDecoder(model_path)
            mapper = ONNXOpMapper(model)
            mapper.paddle_graph.build()
            graph_opt = GraphOptimizer(source_frame="onnx")
            graph_opt.optimize(mapper.paddle_graph)
            _logger.info("Model optimized.")
            onnx2paddle_out_dir = os.path.join(inference_model_path,
                                               'onnx2paddle')
            mapper.paddle_graph.gen_model(onnx2paddle_out_dir)
            _logger.info("Successfully exported Paddle static graph model!")
            if not disable_feedback:
                ConverterCheck(
                    task="ONNX", time_info=time_info,
                    convert_state="Success").start()
            shutil.move(
                os.path.join(onnx2paddle_out_dir, 'inference_model',
                             'model.pdmodel'),
                os.path.join(inference_model_path, 'model.pdmodel'))
            shutil.move(
                os.path.join(onnx2paddle_out_dir, 'inference_model',
                             'model.pdiparams'),
                os.path.join(inference_model_path, 'model.pdiparams'))
        except:
            _logger.info(
                "[ERROR] x2paddle threw an exception, you can ask for help at: https://github.com/PaddlePaddle/X2Paddle/issues"
            )
            sys.exit(1)

        paddle.enable_static()
        val_program, feed_target_names, fetch_targets = paddle.static.load_inference_model(
            os.path.join(inference_model_path, 'model'), exe)
        _logger.info('Loaded model from: {}'.format(inference_model_path))
        return val_program, feed_target_names, fetch_targets
