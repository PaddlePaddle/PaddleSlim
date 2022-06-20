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

import re
import numpy as np
from .parse_ops import get_key_from_op
__all__ = ["get_data_from_tables", "get_features_from_paramkey"]


def cal_flops_params(op_type, cin, cout, kernel=1, h=1, w=1):
    # cin: weight[1]
    if 'conv' in op_type:
        params = cout * (kernel * kernel * cin + 1)
        flops = 2 * kernel * kernel * h * w * cin * cout
        return flops, params
    elif "fc" in op_type:
        flops = 2 * cin * cout
        params = (cin + 1) * cout
        return flops, params


def get_data_from_tables(table_dict, op_type, data_type='fp32'):
    data = []
    for param_key in table_dict:
        cur_type = param_key.split()[0]
        if op_type == cur_type:
            features = get_features_from_paramkey(param_key, op_type, data_type)
            if features == None:
                continue
            # only support bs=1 now
            if features[0] != 1:
                continue
            features.append(table_dict[param_key])
            data.append(features)
    return np.array(data)


def get_features_from_paramkey(param_key, op_type, data_type):
    """Get op's parameters according to the key of latency table
    """
    features = None
    if 'conv2d' in op_type:
        if data_type == 'fp16':
            quant_bits = 'bit_length=16'
        elif data_type == 'int8':
            quant_bits = 'bit_length=8'
        else:
            quant_bits = 'bit_length=None'
        if quant_bits not in param_key:
            return None

        weight = re.search(r'weight=(\(\d*, -?\d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')
        outputs = re.search(r'out=(\(-*\d*, \d*, -?\d*, -?\d*\))',
                            param_key).group().split('=')[-1].strip(
                                '('
                                ')').split(', ')
        batchsize = int(outputs[0])
        cout = int(outputs[1])
        cin = int(weight[1])
        kernel = int(weight[2])
        out_h = int(outputs[2])
        out_w = int(outputs[3])
        stride = int(re.search(r'stride=\d*', param_key).group().split('=')[1])
        group = int(re.search(r'group=\d*', param_key).group().split('=')[1])
        pad = int(re.search(r'pad=\d', param_key).group().split('=')[1])
        flops, params = cal_flops_params('conv', cin, cout, kernel, out_h,
                                         out_w)

        if data_type == 'fp32':
            inputs = re.search(r'in=(\(-*\d*, \d*, -?\d*, -?\d*\))',
                               param_key).group().split('=')[-1].strip(
                                   '('
                                   ')').split(', ')
            in_c = int(inputs[1])
            in_h = int(inputs[2])
            in_w = int(inputs[3])

            features = [
                batchsize, in_c, cout, kernel, group, stride, pad, in_h * in_w,
                out_h * out_w
            ]
        else:
            features = [
                batchsize, cin, cout, kernel, group, stride, pad, out_h * out_w,
                flops, params
            ]

    elif 'matmul' in op_type:
        X = re.search(r'X=(\((-?\d+,* *)+\))',
                      param_key).group().split('=')[-1].strip('('
                                                              ')').split(', ')
        Y = re.search(r'Y=(\((-?\d+,* *)+\))',
                      param_key).group().split('=')[-1].strip('('
                                                              ')').split(', ')

        a = int(X[0])
        b = int(Y[0])
        c = int(Y[1])
        flops, params = cal_flops_params('fc', b, c)

        features = [a, b, c, flops, params]

    elif ('batch_norm' in op_type or 'layer_norm' in op_type):
        inputs = re.search(r'in=(\((-?\d+,* *)+\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')

        features = [0, 0, 0, 0]
        for i in range(len(inputs)):
            if inputs[i] == '':
                continue
            features[i] = int(inputs[i])

    elif 'pool2d' in op_type:

        inputs = re.search(r'in=(\(-*\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')
        outputs = re.search(r'out=(\(-*\d*, \d*, \d*, \d*\))',
                            param_key).group().split('=')[-1].strip(
                                '('
                                ')').split(', ')
        batchsize = int(inputs[0])
        cin = int(inputs[1])
        in_h = int(inputs[2])
        in_w = int(inputs[3])
        out_h = int(outputs[2])
        out_w = int(outputs[3])
        kernel = int(
            re.search(r'kernel=\d*x*\d*', param_key).group().split('x')[-1])
        flag_global = int(
            re.search(r'flag_global=\d', param_key).group().split('=')[-1])
        if flag_global:
            kernel = in_h
        stride = int(re.search(r'stride=\d', param_key).group().split('=')[-1])
        pad = int(re.search(r'pad=\d', param_key).group().split('=')[-1])
        flag_type = 1 if 'type=avg' in param_key else 0

        features = [
            batchsize, cin, kernel, stride, pad, in_h * in_w, out_h * out_w,
            flag_type
        ]

    elif ('reshape' in op_type or 'scale' in op_type):
        inputs = re.search(r'in=(\((-?\d+,* *)+\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(',')
        outputs = re.search(r'out=(\((-?\d+,* *)+\))',
                            param_key).group().split('=')[-1].strip(
                                '('
                                ')').split(',')

        # inputs[4], ouputs[4]/[5]
        features = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(inputs)):
            if inputs[i] == '':
                continue
            features[i] = int(inputs[i])
        for i in range(len(outputs)):
            if outputs[i] == '':
                continue
            features[i + 4] = int(outputs[i])

    elif ('hard_swish' in op_type or 'relu' in op_type or
          'leaky_relu' in op_type or 'tanh' in op_type or 'swish' in op_type or
          'softmax' in op_type or 'hard_sigmoid' in op_type or
          'sigmoid' in op_type or 'gelu' in op_type or 'clip' in op_type or
          'shape' in op_type or 'interp_v2' in op_type or 'sqrt' in op_type):

        inputs = re.search(r'in=(\((-?\d+,* *)+\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')

        # N, C, H, W
        features = [0, 0, 0, 0]
        for i in range(len(inputs)):
            features[i] = int(inputs[i])

    elif 'transpose' in op_type:
        inputs = re.search(r'in=(\((-?\d+,* *)+\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')

        # inputs[4]/[5]
        features = [0, 0, 0, 0, 0]
        for i in range(len(inputs)):
            features[i] = int(inputs[i])

    elif 'elementwise' in op_type:
        X = re.search(r'X=\((-?\d+,* *)+\)',
                      param_key).group().split('=')[-1].strip('('
                                                              ')').split(',')

        Y = re.search(r'Y=\((-?\d+,* *)+\)',
                      param_key).group().split('=')[-1].strip('('
                                                              ')').split(',')
        # X[0] X[1] X[2] X[3] Y[1] Y[2] Y[3]
        features = [0, 0, 0, 0, 0, 0, 0]
        for i in range(len(X)):
            if X[i] == '':
                continue
            features[i] = int(X[i])
        for i in range(0, len(Y)):
            if len(Y) == 4 and i == 0:
                continue
            if Y[i] == '':
                continue
            features[i + 3] = int(Y[i])

    elif 'concat' in op_type:
        inputs = re.search(r'in=(\((-?\d+,* *)+\))+',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(')(')

        channels = []
        for ins in inputs:
            channels.append(int(ins.split(', ')[1]))
        # bs, hw, c1,c2,c3,c4,c5,c6,c7,c8,c9
        features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        input1 = inputs[0].split(', ')
        features[0] = int(input1[0])
        if len(input1) == 3:
            features[1] = int(input1[2])
        else:
            features[1] = int(input1[2]) * int(input1[3])

        for i in range(len(channels)):
            features[i + 2] = channels[i]

    elif 'yolo_box' in op_type:
        outputs = re.search(r'out=(\(-?\d*, \d*, \d*\))',
                            param_key).group().split('=')[-1].strip(
                                '('
                                ')').split(', ')
        inputs = re.search(r'in=(\(-?\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')
        batchsize = int(inputs[0])
        cin = int(inputs[1])
        h = int(inputs[2])
        w = int(inputs[3])
        cout = int(outputs[1])
        class_num = int(
            re.search(r'class_num=\d*', param_key).group().split('=')[-1])

        features = [batchsize, cin, h * w, cout, class_num]

    elif 'prior_box' in op_type:
        inputs = re.search(r'in=\((-?\d+,* *)+\)',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(',')
        batchsize = int(inputs[0])
        cin = int(inputs[1])
        h = int(inputs[2])
        w = int(inputs[3])

        features = [batchsize, cin, h, w]

    elif 'slice' in op_type:
        inputs = re.search(r'in=\((-?\d+,* *)+\)',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(',')
        features = [0, 0, 0, 0]
        for i in range(len(inputs)):
            if inputs[i] == '':
                continue
            features[i] = int(inputs[i])

    elif 'exp' in op_type:
        inputs = re.search(r'in=\((-?\d+,* *)+\)',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(',')
        features = [0, 0, 0, 0]
        for i in range(len(inputs)):
            if inputs[i] == '':
                continue
            features[i] = int(inputs[i])

    elif 'dropout' in param_key:
        inputs = re.search(r'in=\((-?\d+,* *)+\)',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(',')

        features = [0, 0, 0, 0]
        for i in range(len(inputs)):
            if inputs[i] == '':
                continue
            features[i] = int(inputs[i])

    elif 'shuffle_channel' in op_type:
        inputs = re.search(r'in=(\(-*\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')
        batchsize = int(inputs[0])
        cin = int(inputs[1])
        in_h = int(inputs[2])
        in_w = int(inputs[3])
        group = int(re.search(r'group=\d*', param_key).group().split('=')[1])

        features = [batchsize, cin, in_h, in_w, group]

    elif 'split' in op_type:
        inputs = re.search(r'in=(\(-*\d*, \d*, -?\d*, -?\d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')
        batchsize = int(inputs[0])
        cin = int(inputs[1])
        in_h = int(inputs[2])
        in_w = int(inputs[2])

        features = [batchsize, cin, in_h, in_w]

    elif 'squeeze' in op_type:
        inputs = re.search(r'in=\((-?\d+,* *)+\)',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(',')
        features = [0, 0, 0, 0]
        for i in range(len(inputs)):
            if inputs[i] == '':
                continue
            features[i] = int(inputs[i])

    elif 'flatten_contiguous_range' in op_type:
        inputs = re.search(r'in=(\(-?\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')

        features = [
            int(inputs[0]), int(inputs[1]), int(inputs[2]), int(inputs[3])
        ]

    elif ('calib' in op_type or 'floor' in op_type):
        inputs = re.search(r'in=\((-?\d+,* *)+\)',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(',')
        outputs = re.search(r'out=\((-?\d+,* *)+\)',
                            param_key).group().split('=')[-1].strip(
                                '('
                                ')').split(',')
        # inputs[4] outputs[4]
        features = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(inputs)):
            features[i] = int(inputs[i])
        for i in range(len(outputs)):
            features[i + 4] = int(outputs[i])

    elif 'uniform_random' in op_type:
        shape = re.search(r'shape=\[(-?\d+,* *)+\]',
                          param_key).group().split('=')[-1].strip(
                              '['
                              ']').split(',')
        features = [0, 0, 0, 0]
        for i in range(len(shape)):
            if shape[i] == '':
                continue
            features[i] = int(shape[i])

    elif 'arg_max' in op_type:
        inputs = re.search(r'in=\((-?\d+,* *)+\)',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(',')
        outputs = re.search(r'out=\((-?\d+,* *)+\)',
                            param_key).group().split('=')[-1].strip(
                                '('
                                ')').split(',')

        # inputs[4], outputs[4]
        features = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(inputs)):
            if inputs[i] == '':
                continue
            features[i] = int(inputs[i])
        for i in range(len(outputs)):
            if outputs[i] == '':
                continue
            features[i + 4] = int(outputs[i])

    elif 'fill_constant_batch_size_like' in op_type:
        inputs = re.search(r'in=\((-?\d+,* *)+\)',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(',')
        outputs = re.search(r'out=\((-?\d+,* *)+\)',
                            param_key).group().split('=')[-1].strip(
                                '('
                                ')').split(',')

        # inputs[4], outputs[4]
        features = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(inputs)):
            if inputs[i] == '':
                continue
            features[i] = int(inputs[i])
        for i in range(len(outputs)):
            if outputs[i] == '':
                continue
            features[i + 4] = int(outputs[i])

    elif op_type == 'rnn':
        inputs = re.search(r'in=\((-?\d+,* *)+\)',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(',')
        inputs[0], inputs[1] = inputs[1], inputs[0]
        outputs = re.search(r'out=\((-?\d+,* *)+\)',
                            param_key).group().split('=')[-1].strip(
                                '('
                                ')').split(',')
        outputs[0], outputs[1] = outputs[1], outputs[0]

        # inputs[3], outputs[3]
        features = [0, 0, 0, 0, 0, 0]
        for i in range(len(inputs)):
            if inputs[i] == '':
                continue
            features[i] = int(inputs[i])
        for i in range(len(outputs)):
            if outputs[i] == '':
                continue
            features[i + 3] = int(outputs[i])

    return features
