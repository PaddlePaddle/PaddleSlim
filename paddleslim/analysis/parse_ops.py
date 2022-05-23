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

__all__ = ["get_key_from_op"]


def get_key_from_op(op):
    """Construct key of latency table according to the info of graph's op
    """
    param_key = ''
    op_type = op.type()

    if op_type == 'sparse_conv2d':
        out_shape = op.all_outputs()[0].shape()
        in_shape = op.inputs('Input')[0].shape()
        if in_shape:
            weight_shape = (out_shape[1], in_shape[1], 1, 1)
        else:
            weight_shape = (out_shape[1], -1, 1, 1)
        NonZeroWeights = op.inputs('NonZeroWeights')[0].shape()[0]

        stride = op.attr('strides')[1]
        padding = op.attr('paddings')[1]
        groups = op.attr('groups')
        dilation = op.attr('dilations')[1]
        quant = op.attr('enable_int8')
        bit_length = op.attr('bit_length')

        param_key = f'{op_type} in={in_shape} weight={weight_shape} out={out_shape} pad={padding} stride={stride} group={groups} dilation={dilation} quant={quant} bit_length={bit_length} NonZeroWeights={NonZeroWeights}'

    elif 'conv2d' in op_type:
        out_shape = op.all_outputs()[0].shape()
        in_shape = op.all_inputs()[-1].shape()
        in_name = op.all_inputs()[1].name()
        weight_shape = op.all_inputs()[-2].shape()
        weight_shape = (out_shape[1], weight_shape[1], weight_shape[2],
                        weight_shape[3])

        stride = op.attr('strides')[1]
        padding = op.attr('paddings')[1]
        groups = op.attr('groups')
        dilation = op.attr('dilations')[1]
        quant = op.attr('enable_int8')
        bit_length = op.attr('bit_length')
        if op.attr(in_name + '_fp16') == 'fp16':
            quant = True
            bit_length = 16

        param_key = f'{op_type} in={in_shape} weight={weight_shape} out={out_shape} pad={padding} stride={stride} group={groups} dilation={dilation} quant={quant} bit_length={bit_length}'

    elif op_type == 'matmul' or op_type == 'matmul_v2':
        X = op.all_inputs()[0].shape()
        Y = op.all_inputs()[1].shape()
        out_shape = op.all_outputs()[0].shape()
        quant = op.attr('enable_int8')
        bit_length = op.attr('bit_length')

        param_key = f'{op_type} X={X} Y={Y} out={out_shape} quant={quant} bit_length={bit_length}'

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
            'hard_sigmoid', 'sigmoid', 'gelu', 'clip', 'shape', 'sqrt'
    ] or 'transpose' in op_type or 'interp_v2' in op_type:
        in_shape = op.all_inputs()[-1].shape()
        out_shape = op.all_outputs()[0].shape()

        param_key = f'{op_type} in={in_shape} out={out_shape}'

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
        X = ""
        for x in data:
            X += f"{x.shape()}"
        axis = op.attr('axis')
        out_shape = op.all_outputs()[0].shape()

        param_key = f'{op_type} in={X} axis={axis} out={out_shape}'

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

        param_key = f'{op_type} X={in_shape1} Y={in_shape2}  out={out_shape}'

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

    elif op_type == 'arg_max':
        in_shape = op.all_inputs()[-1].shape()
        out_shape = op.all_outputs()[0].shape()
        axis = op.attr('axis')

        param_key = f'{op_type} in={in_shape} axis={axis}  out={out_shape}'

    elif op_type == 'fill_constant_batch_size_like':
        in_shape = op.all_inputs()[-1].shape()
        out_shape = op.all_outputs()[0].shape()
        shape = op.attr('shape')
        param_key = f'{op_type} in={in_shape} shape={shape}  out={out_shape}'

    elif op_type == 'rnn':
        out_shape = op.all_outputs()[1].shape()
        in_shape = op.all_inputs()[0].shape()
        param_key = f'{op_type} in={in_shape} out={out_shape}'

    elif op_type in ['feed', 'fetch']:
        pass

    else:
        param_key = None

    return param_key
