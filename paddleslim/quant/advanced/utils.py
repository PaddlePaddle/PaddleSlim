# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numpy as np
from sklearn.cluster import KMeans


def k_means(weight, n_clusters, init='k-means++', max_iter=300):
    org_shape = weight.shape
    weight = paddle.to_tensor(weight)
    weight = paddle.reshape(weight, [-1, 1])
    if n_clusters > weight.size:
        n_clusters = weight.size

    k_means = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=10,
        algorithm='lloyd',
        max_iter=max_iter)
    k_means.fit(weight)

    centroids = k_means.cluster_centers_
    labels = k_means.labels_
    labels = labels.reshape(org_shape)
    return paddle.to_tensor(centroids.flatten()), paddle.to_tensor(labels)


def compute_scales(x, method='abs_max'):
    if method == 'abs_max':
        quant_scale = float(paddle.max(paddle.abs(x.flatten())))
        quant_scale = 1e-8 if quant_scale == 0.0 else quant_scale
    elif method == 'avg':
        quant_scale = paddle.abs(x.reshape((x.shape[0], -1)))
        quant_scale = paddle.mean(paddle.max(quant_scale, axis=(1)))
    elif method == 'abs_max_channel_wise':
        reduce_axis = tuple([i for i in range(len(x.shape)) if i != 1])
        quant_scale = paddle.max(paddle.abs(x), axis=reduce_axis)
        quant_scale = paddle.where(quant_scale == paddle.to_tensor(
            0, dtype=x.dtype),
                                   paddle.to_tensor(1e-8, dtype=x.dtype),
                                   quant_scale)
    return quant_scale


def find_parent_layer_and_sub_name(model, name):
    last_idx = 0
    idx = 0
    parent_layer = model
    while idx < len(name):
        if name[idx] == '.':
            sub_name = name[last_idx:idx]
            if hasattr(parent_layer, sub_name):
                parent_layer = getattr(parent_layer, sub_name)
                last_idx = idx + 1
        idx += 1
    sub_name = name[last_idx:idx]
    return parent_layer, sub_name


def get_ln_linear_info(ln_linear_list, norm_flag, linear_flag, fused_qkv,
                       llama_ffn, skip_norm_list):
    # ln_linear_dict: {layer_norm_0: [linear_0, linear_1, linear_2]}
    ln_linear_dict = {}
    # linear_ln_dict: {linear_0: layer_norm_0, linear_1: layer_norm_0}
    linear_ln_dict = {}
    for i in range(len(ln_linear_list)):
        layer_name = ln_linear_list[i]
        if norm_flag in layer_name and layer_name not in skip_norm_list:
            if i < len(ln_linear_list) - 1:
                if not fused_qkv:
                    if linear_flag in ln_linear_list[i +
                                                     1] and linear_flag in ln_linear_list[i + 2] and linear_flag in ln_linear_list[i + 3] and int(
                                                         layer_name.split('_')
                                                         [-1]) % 2 == 0:
                        ln_linear_dict[layer_name] = [
                            ln_linear_list[i + 1], ln_linear_list[i + 2],
                            ln_linear_list[i + 3]
                        ]
                        linear_ln_dict[ln_linear_list[i + 1]] = layer_name
                        linear_ln_dict[ln_linear_list[i + 2]] = layer_name
                        linear_ln_dict[ln_linear_list[i + 3]] = layer_name
                    if linear_flag in ln_linear_list[i + 1] and int(
                            layer_name.split('_')[-1]) % 2 != 0:
                        if llama_ffn:
                            ln_linear_dict[layer_name] = [
                                ln_linear_list[i + 1], ln_linear_list[i + 2]
                            ]
                            linear_ln_dict[ln_linear_list[i + 1]] = layer_name
                            linear_ln_dict[ln_linear_list[i + 2]] = layer_name
                        else:
                            ln_linear_dict[layer_name] = [ln_linear_list[i + 1]]
                            linear_ln_dict[ln_linear_list[i + 1]] = layer_name
                else:
                    if linear_flag in ln_linear_list[i + 1]:
                        ln_linear_dict[layer_name] = [ln_linear_list[i + 1]]
                        linear_ln_dict[ln_linear_list[i + 1]] = layer_name
    return ln_linear_dict, linear_ln_dict
