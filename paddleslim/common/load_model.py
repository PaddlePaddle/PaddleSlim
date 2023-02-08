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
import pkg_resources as pkg
import paddle

from . import get_logger
_logger = get_logger(__name__, level=logging.INFO)

__all__ = [
    'load_inference_model', 'get_model_dir', 'load_onnx_model', 'export_onnx'
]


def load_inference_model(path_prefix,
                         executor,
                         model_filename=None,
                         params_filename=None):
    # Load onnx model to Inference model.
    if path_prefix.endswith('.onnx'):
        inference_program, feed_target_names, fetch_targets = load_onnx_model(
            path_prefix)
        return [inference_program, feed_target_names, fetch_targets]
    # Load Inference model.
    # TODO: clean code
    if model_filename is not None and model_filename.endswith('.pdmodel'):
        model_name = '.'.join(model_filename.split('.')[:-1])
        assert os.path.exists(
            os.path.join(path_prefix, model_name + '.pdmodel')
        ), 'Please check {}, or fix model_filename parameter.'.format(
            os.path.join(path_prefix, model_name + '.pdmodel'))
        assert os.path.exists(
            os.path.join(path_prefix, model_name + '.pdiparams')
        ), 'Please check {}, or fix params_filename parameter.'.format(
            os.path.join(path_prefix, model_name + '.pdiparams'))
        model_path_prefix = os.path.join(path_prefix, model_name)
        [inference_program, feed_target_names,
         fetch_targets] = (paddle.static.load_inference_model(
             path_prefix=model_path_prefix, executor=executor))
    elif model_filename is not None and params_filename is not None:
        [inference_program, feed_target_names,
         fetch_targets] = (paddle.static.load_inference_model(
             path_prefix=path_prefix,
             executor=executor,
             model_filename=model_filename,
             params_filename=params_filename))
    else:
        model_name = '.'.join(model_filename.split('.')
                              [:-1]) if model_filename is not None else 'model'
        if os.path.exists(os.path.join(path_prefix, model_name + '.pdmodel')):
            model_path_prefix = os.path.join(path_prefix, model_name)
            [inference_program, feed_target_names,
             fetch_targets] = (paddle.static.load_inference_model(
                 path_prefix=model_path_prefix, executor=executor))
        else:
            [inference_program, feed_target_names,
             fetch_targets] = (paddle.static.load_inference_model(
                 path_prefix=path_prefix, executor=executor))

    return [inference_program, feed_target_names, fetch_targets]


def get_model_dir(model_dir, model_filename, params_filename):
    if model_dir.endswith('.onnx'):
        updated_model_dir = model_dir.rstrip().rstrip('.onnx') + '_infer'
    else:
        updated_model_dir = model_dir.rstrip('/')

    if model_filename == None:
        updated_model_filename = 'model.pdmodel'
    else:
        updated_model_filename = model_filename

    if params_filename == None:
        updated_params_filename = 'model.pdiparams'
    else:
        updated_params_filename = params_filename

    if params_filename is None and model_filename is not None:
        raise NotImplementedError(
            "NOT SUPPORT parameters saved in separate files. Please convert it to single binary file first."
        )
    return updated_model_dir, updated_model_filename, updated_params_filename


def load_onnx_model(model_path,
                    disable_feedback=False,
                    enable_onnx_checker=True):
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
        assert os.path.exists(
            model_path), 'Not found `{}`, please check model path.'.format(
                model_path)
        try:
            import x2paddle
            version = x2paddle.__version__
            v0, v1, v2 = version.split('.')
            version_sum = int(v0) * 100 + int(v1) * 10 + int(v2)
            if version_sum != 139:
                _logger.warning(
                    "x2paddle==1.3.9 is required, please use \"pip install x2paddle==1.3.9\"."
                )
                os.system('python -m pip install -U x2paddle==1.3.9')
        except:
            os.system('python -m pip install -U x2paddle==1.3.9')
        # check onnx installation and version
        try:
            pkg.require('onnx')
            import onnx
            version = onnx.version.version
            v0, v1, v2 = version.split('.')
            version_sum = int(v0) * 100 + int(v1) * 10 + int(v2)
            if version_sum < 160:
                _logger.error(
                    "onnx>=1.6.0 is required, please use \"pip install onnx\".")
        except:
            from pip._internal import main
            main(['install', 'onnx==1.12.0'])

        from x2paddle.decoder.onnx_decoder import ONNXDecoder
        from x2paddle.op_mapper.onnx2paddle.onnx_op_mapper import ONNXOpMapper
        from x2paddle.optimizer.optimizer import GraphOptimizer
        from x2paddle.utils import ConverterCheck
        time_info = int(time.time())
        if not disable_feedback:
            ConverterCheck(
                task="ONNX", time_info=time_info,
                convert_state="Start").start()
        # support distributed convert model
        model_idx = paddle.distributed.get_rank(
        ) if paddle.distributed.get_world_size() > 1 else 0
        try:
            _logger.info("Now translating model from onnx to paddle.")
            model = ONNXDecoder(model_path, enable_onnx_checker)
            mapper = ONNXOpMapper(model)
            mapper.paddle_graph.build()
            graph_opt = GraphOptimizer(source_frame="onnx")
            graph_opt.optimize(mapper.paddle_graph)
            _logger.info("Model optimized.")
            onnx2paddle_out_dir = os.path.join(
                inference_model_path, 'onnx2paddle_{}'.format(model_idx))
            mapper.paddle_graph.gen_model(onnx2paddle_out_dir)
            _logger.info("Successfully exported Paddle static graph model!")
            if not disable_feedback:
                ConverterCheck(
                    task="ONNX", time_info=time_info,
                    convert_state="Success").start()
        except Exception as e:
            _logger.warning(e)
            _logger.error(
                "x2paddle threw an exception, you can ask for help at: https://github.com/PaddlePaddle/X2Paddle/issues"
            )
            sys.exit(1)

        if paddle.distributed.get_rank() == 0:
            shutil.move(
                os.path.join(onnx2paddle_out_dir, 'inference_model',
                             'model.pdmodel'),
                os.path.join(inference_model_path, 'model.pdmodel'))
            shutil.move(
                os.path.join(onnx2paddle_out_dir, 'inference_model',
                             'model.pdiparams'),
                os.path.join(inference_model_path, 'model.pdiparams'))
            load_model_path = inference_model_path
        else:
            load_model_path = os.path.join(onnx2paddle_out_dir,
                                           'inference_model')

        paddle.enable_static()
        val_program, feed_target_names, fetch_targets = paddle.static.load_inference_model(
            os.path.join(load_model_path, 'model'), exe)
        _logger.info('Loaded model from: {}'.format(load_model_path))
        # Clean up the file storage directory
        shutil.rmtree(
            os.path.join(inference_model_path, 'onnx2paddle_{}'.format(
                model_idx)))
        return val_program, feed_target_names, fetch_targets


def export_onnx(model_dir,
                model_filename=None,
                params_filename=None,
                save_file_path='output.onnx',
                opset_version=13,
                deploy_backend='tensorrt'):
    if not model_filename:
        model_filename = 'model.pdmodel'
    if not params_filename:
        params_filename = 'model.pdiparams'
    try:
        import paddle2onnx
        version = paddle2onnx.__version__
        if version < '1.0.1':
            os.system('python -m pip install -U paddle2onnx==1.0.3')
    except:
        from pip._internal import main
        main(['install', 'paddle2onnx==1.0.3'])
    import paddle2onnx
    paddle2onnx.command.c_paddle_to_onnx(
        model_file=os.path.join(model_dir, model_filename),
        params_file=os.path.join(model_dir, params_filename),
        save_file=save_file_path,
        opset_version=opset_version,
        enable_onnx_checker=True,
        deploy_backend=deploy_backend,
        calibration_file=os.path.join(
            save_file_path.rstrip(os.path.split(save_file_path)[-1]),
            'calibration.cache'))
    _logger.info('Convert model to ONNX: {}'.format(save_file_path))
