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
import sklearn
__all__ = [
    "save_cls_model", "save_det_model", "save_seg_model", "nearest_interpolate",
    "load_predictor", "dowload_tools"
]


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
        exe = paddle.fluid.Executor(paddle.fluid.CPUPlace())
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
            activation_bits=8)

        model_file = os.path.join(quantize_model_path, '__model__')
        param_file = os.path.join(quantize_model_path, '__params__')

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


def save_seg_model(model, input_shape, save_dir, data_type):
    if data_type == 'fp32':
        paddle.jit.save(
            model,
            path=os.path.join(save_dir, 'fp32model'),
            input_spec=[
                paddle.static.InputSpec(
                    shape=input_shape, dtype='float32', name='x'),
            ])
        model_file = os.path.join(save_dir, 'fp32model.pdmodel')
        param_file = os.path.join(save_dir, 'fp32model.pdiparams')

    else:
        save_dir = os.path.join(save_dir, 'int8model')
        quant_config = {
            'weight_preprocess_type': None,
            'activation_preprocess_type': None,
            'weight_quantize_type': 'channel_wise_abs_max',
            'activation_quantize_type': 'moving_average_abs_max',
            'weight_bits': 8,
            'activation_bits': 8,
            'dtype': 'int8',
            'window_size': 10000,
            'moving_rate': 0.9,
            'quantizable_layer_type': ['Conv2D', 'Linear'],
        }
        quantizer = paddleslim.QAT(config=quant_config)
        quantizer.quantize(model)
        quantizer.save_quantized_model(
            model,
            save_dir,
            input_spec=[
                paddle.static.InputSpec(
                    shape=input_shape, dtype='float32')
            ])

        model_file = f'{save_dir}.pdmodel'
        param_file = f'{save_dir}.pdiparams'

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
        if distance(features, data_features[i]) < dist:
            idx = i
    return latency[idx]


def dowload_predictor(op_dir, op):
    """Dowload op predictors' model file
        
        Args:
            op_dir(str): the dowload path of op predictor. Actually, it's the hardware information. 
            op(str): the op type.
        Returns:
            op_path: The path of the file.
        """
    if not os.path.exists(op_dir):
        os.makedirs(op_dir)

    op_path = os.path.join(op_dir, op + '_predictor.pkl')
    if not os.path.exists(op_path):
        subprocess.call(
            f'wget -P {op_dir} https://paddlemodels.bj.bcebos.com/PaddleSlim/analysis/{op_path}',
            shell=True)
    return op_path


def load_predictor(op_type, op_dir, data_type='fp32'):
    op = op_type
    if 'conv2d' in op_type:
        op = 'conv2d_' + data_type
    elif 'matmul' in op_type:
        op = 'matmul'

    op_path = dowload_predictor(op_dir, op)
    with open(op_path, 'rb') as f:
        model = pickle.load(f)

    return model


def dowload_tools(platform='mac_intel', lite_version='v2_9'):
    """Dowload tools for LatencyPredictor 
        
        Args:
            platform(str): Operation platform, mac_intel or mac_M1 or ubuntu
            lite_version(str): The version of PaddleLite, v2_9
        Returns:
            opt_path: The path of opt tool to convert a paddle model to an optimized pbmodel that fuses operators.
        """
    opt_name = '_'.join(['opt', platform, lite_version])
    opt_path = os.path.join('./tools', opt_name)
    if not os.path.exists(opt_path):
        subprocess.call(
            f'wget -P ./tools https://paddlemodels.bj.bcebos.com/PaddleSlim/analysis/{opt_name}',
            shell=True)
        subprocess.call(f'chmod +x {opt_path}', shell=True)
    return opt_path
