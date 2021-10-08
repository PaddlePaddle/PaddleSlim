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

import paddle
import paddleslim
__all__ = [
    "get_key_from_op", "save_cls_model", "save_det_model", "save_seg_model"
]


def get_key_from_op(op):
    """Construct key of latency table according to the info of graph's op
    """
    param_key = ''
    op_type = op.type()

    if 'conv2d' in op_type:
        out_shape = op.all_outputs()[0].shape()
        in_shape = op.all_inputs()[-1].shape()
        weight_shape = op.all_inputs()[-2].shape()
        kernel = weight_shape[2]
        stride = op.attr('strides')[1]
        padding = op.attr('paddings')[1]
        groups = op.attr('groups')
        dilation = op.attr('dilations')[1]
        int8 = op.attr('enable_int8')
        bit_length = op.attr('bit_length')

        param_key = f'{op_type} in={in_shape} weight={weight_shape} out={out_shape} pad={padding} stride={stride} group={groups} dilation={dilation} quant={int8} bit_length={bit_length}'

    elif op_type == 'matmul' or op_type == 'matmul_v2':
        X = op.all_inputs()[0].shape()
        Y = op.all_inputs()[1].shape()
        out_shape = op.all_outputs()[0].shape()
        int8 = op.attr('enable_int8')
        bit_length = op.attr('bit_length')

        param_key = f'{op_type} X={X} Y={Y} out={out_shape} quant={int8} bit_length={bit_length}'

    elif 'batch_norm' in op_type or 'layer_norm' in op_type:
        out_shape = op.all_outputs()[-1].shape()
        in_shape = op.all_inputs()[-1].shape()

        param_key = f'{op_type} in={in_shape} out={out_shape}'

    elif 'pool2d' in op_type:
        out_shape = op.all_outputs()[0].shape()
        data = op.all_inputs()
        in_shape = data[-1].shape()
        kernel = op.attr('ksize')[1]
        stride = op.attr('strides')[1]
        padding = op.attr('paddings')[1]
        groups = op.attr('groups')
        flag_global = 1 if op.attr('global_pooling') else 0
        if op.attr('adaptive') and out_shape[-1] == 1:
            flag_global = 1
        pooling_type = op.attr('pooling_type')

        param_key = f'{op_type} in={in_shape} out={out_shape} stride={stride} kernel={kernel}x{kernel} pad={padding} flag_global={flag_global} type={pooling_type})'

    elif op_type in [
            'hard_swish', 'relu', 'leaky_relu', 'tanh', 'swish', 'softmax',
            'hard_sigmoid', 'sigmoid', 'gelu', 'clip', 'shape'
    ] or 'transpose' in op_type or 'interp_v2' in op_type:
        in_shape = op.all_inputs()[-1].shape()

        param_key = f'{op_type} in={in_shape}'
        in_shape = op.all_inputs()[-1].shape()

        param_key = f'{op_type} in={in_shape}'

    elif op_type in ['fill_constant', 'range', 'cast'] or 'expand' in op_type:

        param_key = f'{op_type}'

    elif op_type in ['scale'] or 'reshape' in op_type:
        out_shape = op.all_outputs()[0].shape()
        in_shape = op.all_inputs()[0].shape()

        param_key = f'{op_type} in={in_shape} out={out_shape}'

    elif 'elementwise' in op_type:
        out_shape = op.all_outputs()[0].shape()
        x = op.all_inputs()[0].shape()
        y = op.all_inputs()[1].shape()
        axis = op.attr('axis')

        param_key = f'{op_type} X={x} Y={y} axis={axis} out={out_shape}'

    elif op_type == 'concat':
        data = op.all_inputs()
        X = ""
        for x in data:
            X += f"{x.shape()}"
        axis = op.attr('axis')
        out_shape = op.all_outputs()[0].shape()

        param_key = f'{op_type} in={X} axis={axis} out={out_shape}'

    elif op_type == 'yolo_box':
        in_shape = op.all_inputs()[-1].shape()
        out_shape = op.all_outputs()[0].shape()
        class_num = op.attr('class_num')

        param_key = f'{op_type} in={in_shape} out={out_shape} class_num={class_num}'

    elif op_type == 'prior_box':
        in_shape = op.all_inputs()[-1].shape()
        out_shape = op.all_outputs()[0].shape()
        aspect_ratios = op.attr('aspect_ratios')
        max_sizes = op.attr('max_sizes')
        min_sizes = op.attr('min_sizes')

        param_key = f'{op_type} in={in_shape} out={out_shape} aspect_ratios={aspect_ratios} max_sizes={max_sizes} min_sizes={min_sizes}'

    elif op_type == 'slice':
        in_shape = op.all_inputs()[-1].shape()
        axes = op.attr('axes')

        param_key = f'{op_type} in={in_shape} axes={axes}'

    elif op_type == 'stack':
        data = op.all_inputs()
        X = "["
        for x in data:
            X += f"{x.shape()}"
        X += "]"
        axis = op.attr('axis')
        out_shape = op.all_outputs()[0].shape()

        param_key = f'{op_type} X={X} axis={axis} out={out_shape}'

    elif op_type == 'exp':
        in_shape = op.all_inputs()[-1].shape()
        out_shape = op.all_outputs()[0].shape()
        axes = op.attr('axes')
        decrease_axis = op.attr('decrease_axis')
        ends = op.attr('ends')

        param_key = f'{op_type} in={in_shape} out={out_shape} axes={axes} decrease_axis={decrease_axis} ends={ends}'

    elif op_type in ['multiclass_nms3', 'matrix_nms']:
        boxs = op.all_inputs()[0].shape()
        scores = op.all_inputs()[-1].shape()
        keep_top_k = op.attr('keep_top_k')
        nms_top_k = op.attr('nms_top_k')

        param_key = f'{op_type} boxs={boxs} scores={scores} keep_top_k={keep_top_k} nms_top_k={nms_top_k}'

    elif op_type == 'dropout':
        in_shape = op.all_inputs()[0].shape()

        param_key = f'{op_type} in={in_shape}'

    elif op_type == 'fc':
        in_shape = op.all_inputs()[-2].shape()
        weight_shape = op.all_inputs()[-1].shape()
        out_shape = op.all_outputs()[0].shape()

        param_key = f'{op_type} in={in_shape} weight={weight_shape} out={out_shape}'

    elif op_type == 'shuffle_channel':
        in_shape = op.all_inputs()[-1].shape()
        out_shape = op.all_outputs()[0].shape()
        group = op.attr('group')

        param_key = f'{op_type} in={in_shape} group={group}  out={out_shape}'

    elif op_type == 'split':
        in_shape = op.all_inputs()[-1].shape()
        axis = op.attr('axis')
        sections = op.attr('sections')

        param_key = f'{op_type} in={in_shape} axis={axis} sections={sections}'

    elif op_type in ['unsqueeze2', 'squeeze2']:
        in_shape = op.all_inputs()[-1].shape()
        out_shape = op.all_outputs()[0].shape()
        axes = op.attr('axes')

        param_key = f'{op_type} in={in_shape} axes={axes}  out={out_shape}'

    elif op_type == 'flatten_contiguous_range':
        in_shape = op.all_inputs()[-1].shape()
        out_shape = op.all_outputs()[0].shape()
        start_axis = op.attr('start_axis')
        stop_axis = op.attr(' stop_axis')

        param_key = f'{op_type} in={in_shape} start_axis={start_axis} stop_axis={stop_axis} out={out_shape}'

    elif op_type == 'sum':
        in_shape1 = op.all_inputs()[0].shape()
        in_shape2 = op.all_inputs()[1].shape()
        out_shape = op.all_outputs()[0].shape()

        param_key = f'{op_type} in={in_shape1} in={in_shape2}  out={out_shape}'

    elif op_type in ['calib', 'floor']:
        in_shape = op.all_inputs()[-1].shape()
        out_shape = op.all_inputs()[0].shape()

        param_key = f'{op_type} in={in_shape} out={out_shape}'

    elif op_type == 'uniform_random':
        shape = op.attr('shape')

        param_key = f'{op_type} shape={shape}'

    elif op_type == 'greater_equal':
        x = op.all_inputs()[0].shape()
        y = op.all_inputs()[1].shape()
        out_shape = op.all_outputs()[0].shape()

        param_key = f'{op_type} X={x} Y={y} out={out_shape}'

    elif op_type == 'reduce_mean':
        in_shape = op.all_inputs()[-1].shape()
        out_shape = op.all_outputs()[0].shape()
        dim = op.attr('dim')

        param_key = f'{op_type} in={in_shape} out={out_shape} dim={dim}'

    elif 'pad3d' in op_type:
        in_shape = op.all_inputs()[-1].shape()
        out_shape = op.all_outputs()[0].shape()
        paddings = op.attr('paddings')

        param_key = f'{op_type} in={in_shape} out={out_shape} paddings={paddings}'

    elif op_type in ['feed', 'fetch']:
        pass

    else:
        print(op)
        print(op._op)
        raise KeyError(f'The "{op_type}" has never seen.')

    return param_key


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
