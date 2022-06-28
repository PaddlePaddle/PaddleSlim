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

__all__ = ['load_inference_model']


def load_inference_model(path_prefix,
                         executor,
                         model_filename=None,
                         params_filename=None):
    if model_filename is not None and params_filename is not None:
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
