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

import os
import paddle
from ...common import load_onnx_model

__all__ = ['load_inference_model', 'get_model_dir']


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
        [inference_program, feed_target_names, fetch_targets] = (
            paddle.static.load_inference_model(
                path_prefix=model_path_prefix, executor=executor))
    elif model_filename is not None and params_filename is not None:
        [inference_program, feed_target_names, fetch_targets] = (
            paddle.static.load_inference_model(
                path_prefix=path_prefix,
                executor=executor,
                model_filename=model_filename,
                params_filename=params_filename))
    else:
        model_name = '.'.join(model_filename.split('.')
                              [:-1]) if model_filename is not None else 'model'
        if os.path.exists(os.path.join(path_prefix, model_name + '.pdmodel')):
            model_path_prefix = os.path.join(path_prefix, model_name)
            [inference_program, feed_target_names, fetch_targets] = (
                paddle.static.load_inference_model(
                    path_prefix=model_path_prefix, executor=executor))
        else:
            [inference_program, feed_target_names, fetch_targets] = (
                paddle.static.load_inference_model(
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
