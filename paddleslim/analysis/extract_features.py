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

            features.append(table_dict[param_key])
            data.append(features)
    return np.array(data)


def get_features_from_paramkey(param_key, op_type, data_type):
    """Get op's parameters according to the key of latency table
    """
    features = []

    if 'conv2d' in param_key:
        flag_quant = 'quant=None' if data_type == 'fp32' else 'quant=True'
        if flag_quant not in param_key:
            return None

        weight = re.search(r'weight=(\(\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')
        if re.search(r'out=(\(-*\d*, \d*, \d*, \d*\))', param_key) == None:
            return None
        outputs = re.search(r'out=(\(-*\d*, \d*, \d*, \d*\))',
                            param_key).group().split('=')[-1].strip(
                                '('
                                ')').split(', ')

        # quant = 32 if 'None' in re.search(r'quant=\w*', param_key).group() else 0

        cout = int(weight[0])
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
            inputs = re.search(r'in=(\(-*\d*, \d*, \d*, \d*\))',
                               param_key).group().split('=')[-1].strip(
                                   '('
                                   ')').split(', ')
            in_c = int(inputs[1])
            in_h = int(inputs[2])
            in_w = int(inputs[3])
            # flops, params = get_flops_params('conv', in_h, in_c, cout, kernel, stride)

            features = [
                in_c, cout, kernel, group, stride, pad, in_h * in_w,
                out_h * out_w
            ]
        else:
            features = [
                cin, cout, kernel, group, stride, pad, out_h * out_w, flops,
                params
            ]

    elif 'matmul' in param_key:
        X = re.search(r'X=(\(-*\d*, \d*\))',
                      param_key).group().split('=')[-1].strip('('
                                                              ')').split(', ')
        Y = re.search(r'Y=(\(\d*, \d*\))',
                      param_key).group().split('=')[-1].strip('('
                                                              ')').split(', ')

        a = int(X[0])
        b = int(Y[0])
        c = int(Y[1])
        flops, params = cal_flops_params('fc', b, c)

        features = [b, c, flops, params]

    elif ('batch_norm' in param_key or 'layer_norm' in param_key):
        inputs = re.search(r'in=(\(-*\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')

        cin = int(inputs[1])
        in_h = int(inputs[2])
        in_w = int(inputs[3])

        features = [cin, in_h, in_w]

    elif 'pool2d' in param_key:

        inputs = re.search(r'in=(\(-*\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')
        outputs = re.search(r'out=(\(-*\d*, \d*, \d*, \d*\))',
                            param_key).group().split('=')[-1].strip(
                                '('
                                ')').split(', ')

        cin = int(inputs[1])
        in_h = int(inputs[2])
        in_w = int(inputs[3])
        out_h = int(outputs[2])
        out_w = int(outputs[3])
        kernel = int(
            re.search(r'kernel=\d*x*\d*', param_key).group().split('x')[-1])
        flag_global = bool(
            re.search(r'flag_global=\d', param_key).group().split('=')[-1])
        if flag_global:
            kernel = in_h
        stride = int(re.search(r'stride=\d', param_key).group().split('=')[-1])

        features = [cin, kernel, stride, in_h * in_w, out_h * out_w]

    elif ('hard_swish' in param_key or 'relu' in param_key or
          'leaky_relu' in param_key or 'tanh' in param_key or
          'swish' in param_key or 'softmax' in param_key or
          'hard_sigmoid' in param_key or 'sigmoid' in param_key or
          'gelu' in param_key or 'clip' in param_key or 'shape' in param_key or
          'transpose' in param_key or 'interp_v2' in param_key):
        inputs = re.search(r'in=(\((-?\d+,* *)+\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')
        #cin, h, w
        cin = int(inputs[1])
        in_h = 0
        in_w = 0
        if len(inputs) == 4:
            in_h = int(inputs[2])
            in_w = int(inputs[3])

        features = [cin, in_h, in_w]

    elif ('reshape' in param_key or 'scale' in param_key):
        inputs = re.search(r'in=(\(-*\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')

        cin = int(inputs[1])
        in_h = int(inputs[2])
        in_w = int(inputs[3])

        features = [cin, in_h, in_w]

    elif 'elementwise' in param_key:
        X = re.search(r'X=\((-?\d+,* *)+\)',
                      param_key).group().split('=')[-1].strip('('
                                                              ')').split(',')

        Y = re.search(r'Y=\((-?\d+,* *)+\)',
                      param_key).group().split('=')[-1].strip('('
                                                              ')').split(',')
        # X[1] X[2] X[3] Y[1] Y[2] Y[3]
        features = [0, 0, 0, 0, 0, 0]
        for i in range(1, len(X)):
            if X[i] == '':
                continue
            features[i - 1] = int(X[i])
        for i in range(0, len(Y)):
            if len(Y) == 4 and i == 0:
                continue
            if Y[i] == '':
                continue
            features[i + 2] = int(Y[i])

    elif 'concat' in param_key:
        inputs = re.search(r'in=(\((-?\d+,* *)+\))+',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(')(')

        channels = []
        for ins in inputs:
            channels.append(int(ins.split(', ')[1]))
        #hw, c1,c2...
        features = [0, 0, 0, 0, 0, 0, 0]
        input1 = inputs[0].split(', ')
        if len(input1) == 3:
            features[0] = int(input1[2])
        else:
            features[0] = int(input1[2]) * int(input1[3])

        for i in range(len(channels)):
            features[i + 1] = channels[i]

    elif 'yolo_box' in param_key:
        outputs = re.search(r'out=(\(-?\d*, \d*, \d*\))',
                            param_key).group().split('=')[-1].strip(
                                '('
                                ')').split(', ')
        inputs = re.search(r'in=(\(-?\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')

        cin = int(inputs[1])
        h = int(inputs[2])
        w = int(inputs[3])
        cout = int(outputs[1])
        class_num = int(
            re.search(r'class_num=\d*', param_key).group().split('=')[-1])

        features = [cin, h * w, cout, class_num]

    elif 'prior_box' in param_key:
        outputs = re.search(r'out=(\(-?\d*, \d*, \d*\))',
                            param_key).group().split('=')[-1].strip(
                                '('
                                ')').split(', ')
        inputs = re.search(r'in=(\(-?\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')

        cin = int(inputs[1])
        h = int(inputs[2])
        w = int(inputs[3])
        cout = int(outputs[1])
        max_sizes = int(
            re.search(r'max_sizes=\d*', param_key).group().split('=')[-1])

        features = [cin, h * w, cout, max_sizes]

    elif 'slice' in param_key:
        inputs = re.search(r'in=(\(-?\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')
        features = [int(inputs[1]), int(inputs[2]), int(inputs[3])]

    # elif 'stack' in param_key and op_type == 'stack':

    #     print(param_key)

    elif 'exp' in param_key:
        inputs = re.search(r'in=(\(-?\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')

        features = [int(inputs[1]), int(inputs[2]), int(inputs[3])]

    elif 'dropout' in param_key:
        inputs = re.search(r'in=(\(-?\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')

        features = [int(inputs[1]), int(inputs[2]), int(inputs[3])]

    elif 'fc' in param_key:
        weight = re.search(r'weight=(\(\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')
        # if re.search(r'out=(\(-*\d*, \d*, \d*, \d*\))', param_key)==None:
        #     continue

        cin = int(weight[0])
        cout = int(weight[1])
        flops, params = cal_flops_params('fc', cin, cout)

        features = [cin, cout, flops, params]

    elif 'shuffle_channel' in param_key:
        inputs = re.search(r'in=(\(-*\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')
        cin = int(inputs[1])
        in_h = int(inputs[2])
        in_w = int(inputs[3])
        group = int(re.search(r'group=\d*', param_key).group().split('=')[1])

        features = [cin, in_h, in_w, group]

    elif 'split' in param_key:
        inputs = re.search(r'in=(\(-*\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')
        cin = int(inputs[1])
        in_h = int(inputs[2])
        in_w = int(inputs[2])

        features = [cin, in_h, in_w]

    elif 'squeeze' in param_key:
        inputs = re.search(r'in=\((-?\d+,* *)+\)',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')
        cin = int(inputs[1])
        in_h = 0
        in_w = 0
        if len(inputs) == 4:
            in_h = int(inputs[2])
            in_w = int(inputs[2])

        features = [cin, in_h, in_w]

    elif 'flatten_contiguous_range' in param_key:
        inputs = re.search(r'in=(\(-?\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')

        features = [int(inputs[1]), int(inputs[2]), int(inputs[3])]

    elif 'sum' in param_key:
        inputs = re.findall(r'(in=\(-?\d*, \d*, \d*, \d*\))', param_key)

        input1 = inputs[0].split('=')[-1].strip('(' ')').split(', ')
        input2 = inputs[1].split('=')[-1].strip('(' ')').split(', ')

    elif ('calib' in param_key or 'floor' in param_key):
        if re.search(r'in=(\(-?\d*, \d*, \d*, \d*\))', param_key) == None:
            return None
        inputs = re.search(r'in=(\(-?\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')
        outputs = re.search(r'out=(\(-?\d*, \d*, \d*, \d*\))',
                            param_key).group().split('=')[-1].strip(
                                '('
                                ')').split(', ')

        features = [
            int(inputs[1]), int(inputs[2]), int(inputs[3]), int(outputs[1]),
            int(outputs[3]), int(outputs[3])
        ]

    elif 'uniform_random' in param_key:
        shape = re.search(r'shape=(\(-?\d*, \d*, \d*, \d*\))',
                          param_key).group().split('=')[-1].strip(
                              '('
                              ')').split(', ')

    elif 'greater_equal' in param_key:
        outputs = re.search(r'out=(\(-?\d*, \d*, \d*\))',
                            param_key).group().split('=')[-1].strip(
                                '('
                                ')').split(', ')

    elif 'reduce_mean' in param_key:
        inputs = re.search(r'in=(\(-?\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')
        outputs = re.search(r'out=(\(-?\d*, \d*, \d*\))',
                            param_key).group().split('=')[-1].strip(
                                '('
                                ')').split(', ')

    elif 'pad3d' in param_key:
        inputs = re.search(r'in=(\(-?\d*, \d*, \d*, \d*\))',
                           param_key).group().split('=')[-1].strip(
                               '('
                               ')').split(', ')
        outputs = re.search(r'out=(\(-?\d*, \d*, \d*\))',
                            param_key).group().split('=')[-1].strip(
                                '('
                                ')').split(', ')

    return features
