# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import numpy as np
import pickle
import paddle
import paddleslim
import subprocess
import time
import requests
import shutil
import logging
from ..common import get_logger
_logger = get_logger(__name__, level=logging.INFO)
__all__ = [
    "save_cls_model", "save_det_model", "nearest_interpolate", "opt_model",
    "load_predictor"
]

PREDICTOR_URL = 'https://paddlemodels.bj.bcebos.com/PaddleSlim/analysis/'


def _get_download(url, fullname):
    # using requests.get method
    fname = os.path.basename(fullname)
    try:
        req = requests.get(url, stream=True)
    except Exception as e:  # requests.exceptions.ConnectionError
        _logger.info("Downloading {} from {} failed with exception {}".format(
            fname, url, str(e)))
        return False

    if req.status_code != 200:
        raise RuntimeError("Downloading from {} failed with code "
                           "{}!".format(url, req.status_code))

    # For protecting download interupted, download to
    # tmp_fullname firstly, move tmp_fullname to fullname
    # after download finished
    tmp_fullname = fullname + "_tmp"
    with open(tmp_fullname, 'wb') as f:
        for chunk in req.iter_content(chunk_size=1024):
            f.write(chunk)

    try:
        shutil.move(tmp_fullname, fullname)
    except:
        shutil.rmtree(tmp_fullname, ignore_errors=True)

    return fullname


def opt_model(opt="paddle_lite_opt",
              model_file='',
              param_file='',
              optimize_out_type='protobuf',
              valid_targets='arm',
              enable_fp16=False):
    assert os.path.exists(model_file) and os.path.exists(
        param_file), f'{model_file} or {param_file} does not exist.'
    save_dir = f'./opt_models_tmp/{os.getpid()}_{time.time()}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    assert optimize_out_type in ['protobuf', 'naive_buffer']
    if optimize_out_type == 'protobuf':
        model_out = save_dir
    else:
        model_out = os.path.join(save_dir, 'model')

    enable_fp16 = str(enable_fp16).lower()

    cmd = f'{opt} --model_file={model_file} --param_file={param_file}  --optimize_out_type={optimize_out_type} --optimize_out={model_out} --valid_targets={valid_targets} --enable_fp16={enable_fp16} --sparse_model=true --sparse_threshold=0.4'
    print(f'commands:{cmd}')
    m = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out = m.communicate()
    print(out, 'opt done!')

    if optimize_out_type == 'protobuf':
        model_out = os.path.join(model_out, 'model')
    else:
        model_out = model_out + '.nb'
    assert os.path.exists(
        model_out
    ), 'There is an error during \'opt\' conversion model, please check the above error message.'
    return model_out


def sample_generator(input_shape, batch_num):
    def __reader__():
        for i in range(batch_num):
            image = np.random.random(input_shape).astype('float32')
            yield image

    return __reader__


def save_cls_model(model, input_shape, save_dir, data_type):
    paddle.jit.save(
        model,
        path=os.path.join(save_dir, 'fp32model'),
        input_spec=[
            paddle.static.InputSpec(
                shape=input_shape, dtype='float32', name='x'),
        ])
    model_file = os.path.join(save_dir, 'fp32model.pdmodel')
    param_file = os.path.join(save_dir, 'fp32model.pdiparams')

    if data_type == 'int8':
        paddle.enable_static()
        exe = paddle.static.Executor(paddle.CPUPlace())
        save_dir = os.path.dirname(model_file)

        quantize_model_path = os.path.join(save_dir, 'int8model')
        if not os.path.exists(quantize_model_path):
            os.makedirs(quantize_model_path)

        paddleslim.quant.quant_post_static(
            executor=exe,
            model_dir=save_dir,
            quantize_model_path=quantize_model_path,
            sample_generator=sample_generator(input_shape, 1),
            model_filename=model_file.split('/')[-1],
            params_filename=param_file.split('/')[-1],
            batch_size=input_shape[0],
            batch_nums=1,
            weight_bits=8,
            activation_bits=8,
            quantizable_op_type=["conv2d", "depthwise_conv2d"],
            onnx_format=False)

        model_file = os.path.join(quantize_model_path, 'model.pdmodel')
        param_file = os.path.join(quantize_model_path, 'model.pdiparams')

    return model_file, param_file


def save_det_model(model,
                   input_shape,
                   save_dir,
                   data_type,
                   det_multi_input=False):
    model.eval()
    if det_multi_input:
        input_spec = [{
            "image": paddle.static.InputSpec(
                shape=input_shape, name='image'),
            "im_shape": paddle.static.InputSpec(
                shape=[input_shape[0], 2], name='im_shape'),
            "scale_factor": paddle.static.InputSpec(
                shape=[input_shape[0], 2], name='scale_factor')
        }]
        data = {
            "image": paddle.randn(
                shape=input_shape, dtype='float32', name='image'),
            "im_shape": paddle.randn(
                shape=[input_shape[0], 2], dtype='float32', name='image'),
            "scale_factor": paddle.ones(
                shape=[input_shape[0], 2], dtype='float32', name='image')
        }
    else:
        input_spec = [{
            "image": paddle.static.InputSpec(
                shape=input_shape, name='image'),
        }]
        data = {
            "image": paddle.randn(
                shape=input_shape, dtype='float32', name='image'),
        }

    if data_type == 'fp32':
        static_model = paddle.jit.to_static(model, input_spec=input_spec)
        paddle.jit.save(
            static_model,
            path=os.path.join(save_dir, 'fp32model'),
            input_spec=input_spec)
        model_file = os.path.join(save_dir, 'fp32model.pdmodel')
        param_file = os.path.join(save_dir, 'fp32model.pdiparams')

    else:
        ptq = paddleslim.dygraph.quant.PTQ()
        quant_model = ptq.quantize(model, fuse=True, fuse_list=None)
        quant_model(data)
        quantize_model_path = os.path.join(save_dir, 'int8model')
        if not os.path.exists(quantize_model_path):
            os.makedirs(quantize_model_path)

        ptq.save_quantized_model(quant_model,
                                 os.path.join(quantize_model_path, 'int8model'),
                                 input_spec)

        model_file = os.path.join(quantize_model_path, 'int8model.pdmodel')
        param_file = os.path.join(quantize_model_path, 'int8model.pdiparams')

    return model_file, param_file


def nearest_interpolate(features, data):
    def distance(x, y):
        x = np.array(x)
        y = np.array(y)
        return np.sqrt(np.sum(np.square(x - y)))

    if len(data) <= 0:
        return None
    data_features = data[:, 0:-1]
    latency = data[:, -1]
    idx = 0
    dist = distance(features, data_features[0])
    for i in range(1, len(data_features)):
        cur_dist = distance(features, data_features[i])
        if cur_dist < dist:
            idx = i
            dist = cur_dist
    return latency[idx]


def download_predictor(op_dir, op):
    """Download op predictors' model file
        
        Args:
            op_dir(str): the path to op predictor. Actually, it's the hardware information. 
            op(str): the op type.
        Returns:
            op_path: The path of the file.
        """
    if not os.path.exists(op_dir):
        os.makedirs(op_dir)

    op_path = os.path.join(op_dir, op + '_predictor.pkl')
    url = PREDICTOR_URL + op_path
    while not (os.path.exists(op_path)):
        if not _get_download(url, op_path):
            time.sleep(1)
            continue

    print('Successfully download {}!'.format(op_path))
    return op_path


def load_predictor(op_type, op_dir, data_type='fp32'):
    op = op_type
    if 'conv2d' in op_type:
        op = f'{op_type}_{data_type}'
    elif 'matmul' in op_type:
        op = 'matmul'

    op_path = download_predictor(op_dir, op)
    with open(op_path, 'rb') as f:
        model = pickle.load(f)
    return model
