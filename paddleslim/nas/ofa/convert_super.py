#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import inspect
import decorator
import logging
import paddle
import numbers
from paddle.fluid.dygraph.nn import Conv2D, Conv2DTranspose, Linear, BatchNorm, InstanceNorm, LayerNorm, Embedding
from .layers import *
from ...common import get_logger

_logger = get_logger(__name__, level=logging.INFO)

__all__ = ['supernet']

WEIGHT_LAYER = ['conv', 'linear', 'embedding']


### TODO: add decorator
class Convert:
    def __init__(self, context):
        self.context = context

    def convert(self, model):
        # search the first and last weight layer, don't change out channel of the last weight layer
        # don't change in channel of the first weight layer
        first_weight_layer_idx = -1
        last_weight_layer_idx = -1
        weight_layer_count = 0
        # NOTE: pre_channel store for shortcut module
        pre_channel = 0
        cur_channel = None
        for idx, layer in enumerate(model):
            cls_name = layer.__class__.__name__.lower()
            if 'conv' in cls_name or 'linear' in cls_name or 'embedding' in cls_name:
                weight_layer_count += 1
                last_weight_layer_idx = idx
                if first_weight_layer_idx == -1:
                    first_weight_layer_idx = idx

        if getattr(self.context, 'channel', None) != None:
            assert len(
                self.context.channel
            ) == weight_layer_count, "length of channel must same as weight layer."

        for idx, layer in enumerate(model):
            if isinstance(layer, Conv2D):
                attr_dict = layer.__dict__
                key = attr_dict['_full_name']

                new_attr_name = [
                    '_stride', '_dilation', '_groups', '_param_attr',
                    '_bias_attr', '_use_cudnn', '_act', '_dtype', '_padding'
                ]

                new_attr_dict = dict()
                new_attr_dict['candidate_config'] = dict()
                self.kernel_size = getattr(self.context, 'kernel_size', None)

                if self.kernel_size != None:
                    new_attr_dict['transform_kernel'] = True

                # if the kernel_size of conv is 1, don't change it.
                #if self.kernel_size and int(attr_dict['_filter_size'][0]) != 1:
                if self.kernel_size and int(attr_dict['_filter_size']) != 1:
                    new_attr_dict['filter_size'] = max(self.kernel_size)
                    new_attr_dict['candidate_config'].update({
                        'kernel_size': self.kernel_size
                    })
                else:
                    new_attr_dict['filter_size'] = attr_dict['_filter_size']

                if self.context.expand:
                    ### first super convolution
                    if idx == first_weight_layer_idx:
                        new_attr_dict['num_channels'] = attr_dict[
                            '_num_channels']
                    else:
                        new_attr_dict[
                            'num_channels'] = self.context.expand * attr_dict[
                                '_num_channels']
                    ### last super convolution
                    if idx == last_weight_layer_idx:
                        new_attr_dict['num_filters'] = attr_dict['_num_filters']
                    else:
                        new_attr_dict[
                            'num_filters'] = self.context.expand * attr_dict[
                                '_num_filters']
                        new_attr_dict['candidate_config'].update({
                            'expand_ratio': self.context.expand_ratio
                        })
                elif self.context.channel:
                    if attr_dict['_groups'] != None and (
                            int(attr_dict['_groups']) ==
                            int(attr_dict['_num_channels'])):
                        ### depthwise conv, if conv is depthwise, use pre channel as cur_channel
                        _logger.warn(
                        "If convolution is a depthwise conv, output channel change" \
                        " to the same channel with input, output channel in search is not used."
                        )
                        cur_channel = pre_channel
                    else:
                        cur_channel = self.context.channel[0]
                    self.context.channel = self.context.channel[1:]
                    if idx == first_weight_layer_idx:
                        new_attr_dict['num_channels'] = attr_dict[
                            '_num_channels']
                    else:
                        new_attr_dict['num_channels'] = max(pre_channel)

                    if idx == last_weight_layer_idx:
                        new_attr_dict['num_filters'] = attr_dict['_num_filters']
                    else:
                        new_attr_dict['num_filters'] = max(cur_channel)
                        new_attr_dict['candidate_config'].update({
                            'channel': cur_channel
                        })
                        pre_channel = cur_channel
                else:
                    new_attr_dict['num_filters'] = attr_dict['_num_filters']
                    new_attr_dict['num_channels'] = attr_dict['_num_channels']

                for attr in new_attr_name:
                    new_attr_dict[attr[1:]] = attr_dict[attr]

                del layer

                if attr_dict['_groups'] == None or int(attr_dict[
                        '_groups']) == 1:
                    ### standard conv
                    layer = Block(SuperConv2D(**new_attr_dict), key=key)
                elif int(attr_dict['_groups']) == int(attr_dict[
                        '_num_channels']):
                    # if conv is depthwise conv, groups = in_channel, out_channel = in_channel,
                    # channel in candidate_config = in_channel_list
                    if 'channel' in new_attr_dict['candidate_config']:
                        new_attr_dict['num_channels'] = max(cur_channel)
                        new_attr_dict['num_filters'] = new_attr_dict[
                            'num_channels']
                        new_attr_dict['candidate_config'][
                            'channel'] = cur_channel
                    new_attr_dict['groups'] = new_attr_dict['num_channels']
                    layer = Block(
                        SuperDepthwiseConv2D(**new_attr_dict), key=key)
                else:
                    ### group conv
                    layer = Block(SuperGroupConv2D(**new_attr_dict), key=key)
                model[idx] = layer

            elif isinstance(layer, BatchNorm) and (
                    getattr(self.context, 'expand', None) != None or
                    getattr(self.context, 'channel', None) != None):
                # num_features in BatchNorm don't change after last weight operators
                if idx > last_weight_layer_idx:
                    continue

                attr_dict = layer.__dict__
                new_attr_name = [
                    '_param_attr', '_bias_attr', '_act', '_dtype', '_in_place',
                    '_data_layout', '_momentum', '_epsilon', '_is_test',
                    '_use_global_stats', '_trainable_statistics'
                ]
                new_attr_dict = dict()
                if self.context.expand:
                    new_attr_dict['num_channels'] = self.context.expand * int(
                        layer._parameters['weight'].shape[0])
                elif self.context.channel:
                    new_attr_dict['num_channels'] = max(cur_channel)
                else:
                    new_attr_dict['num_channels'] = attr_dict['_num_channels']

                for attr in new_attr_name:
                    new_attr_dict[attr[1:]] = attr_dict[attr]

                del layer, attr_dict

                layer = SuperBatchNorm(**new_attr_dict)
                model[idx] = layer

            ### assume output_size = None, filter_size != None
            ### NOTE: output_size != None may raise error, solve when it happend. 
            elif isinstance(layer, Conv2DTranspose):
                attr_dict = layer.__dict__
                key = attr_dict['_full_name']

                new_attr_name = [
                    '_stride', '_dilation', '_groups', '_param_attr',
                    '_padding', '_bias_attr', '_use_cudnn', '_act', '_dtype',
                    '_output_size'
                ]
                assert attr_dict[
                    '_filter_size'] != None, "Conv2DTranspose only support filter size != None now"

                new_attr_dict = dict()
                new_attr_dict['candidate_config'] = dict()
                self.kernel_size = getattr(self.context, 'kernel_size', None)

                if self.kernel_size != None:
                    new_attr_dict['transform_kernel'] = True

                # if the kernel_size of conv transpose is 1, don't change it.
                if self.kernel_size and int(attr_dict['_filter_size'][0]) != 1:
                    new_attr_dict['filter_size'] = max(self.kernel_size)
                    new_attr_dict['candidate_config'].update({
                        'kernel_size': self.kernel_size
                    })
                else:
                    new_attr_dict['filter_size'] = attr_dict['_filter_size']

                if self.context.expand:
                    ### first super convolution transpose
                    if idx == first_weight_layer_idx:
                        new_attr_dict['num_channels'] = attr_dict[
                            '_num_channels']
                    else:
                        new_attr_dict[
                            'num_channels'] = self.context.expand * attr_dict[
                                '_num_channels']
                    ### last super convolution transpose
                    if idx == last_weight_layer_idx:
                        new_attr_dict['num_filters'] = attr_dict['_num_filters']
                    else:
                        new_attr_dict[
                            'num_filters'] = self.context.expand * attr_dict[
                                '_num_filters']
                        new_attr_dict['candidate_config'].update({
                            'expand_ratio': self.context.expand_ratio
                        })
                elif self.context.channel:
                    if attr_dict['_groups'] != None and (
                            int(attr_dict['_groups']) ==
                            int(attr_dict['_num_channels'])):
                        ### depthwise conv_transpose
                        _logger.warn(
                        "If convolution is a depthwise conv_transpose, output channel " \
                        "change to the same channel with input, output channel in search is not used."
                        )
                        cur_channel = pre_channel
                    else:
                        cur_channel = self.context.channel[0]
                    self.context.channel = self.context.channel[1:]
                    if idx == first_weight_layer_idx:
                        new_attr_dict['num_channels'] = attr_dict[
                            '_num_channels']
                    else:
                        new_attr_dict['num_channels'] = max(pre_channel)

                    if idx == last_weight_layer_idx:
                        new_attr_dict['num_filters'] = attr_dict['_num_filters']
                    else:
                        new_attr_dict['num_filters'] = max(cur_channel)
                        new_attr_dict['candidate_config'].update({
                            'channel': cur_channel
                        })
                        pre_channel = cur_channel
                else:
                    new_attr_dict['num_filters'] = attr_dict['_num_filters']
                    new_attr_dict['num_channels'] = attr_dict['_num_channels']

                for attr in new_attr_name:
                    new_attr_dict[attr[1:]] = attr_dict[attr]

                del layer

                if new_attr_dict['output_size'] == []:
                    new_attr_dict['output_size'] = None

                if attr_dict['_groups'] == None or int(attr_dict[
                        '_groups']) == 1:
                    ### standard conv_transpose
                    layer = Block(
                        SuperConv2DTranspose(**new_attr_dict), key=key)
                elif int(attr_dict['_groups']) == int(attr_dict[
                        '_num_channels']):
                    # if conv is depthwise conv, groups = in_channel, out_channel = in_channel,
                    # channel in candidate_config = in_channel_list
                    if 'channel' in new_attr_dict['candidate_config']:
                        new_attr_dict['num_channels'] = max(cur_channel)
                        new_attr_dict['num_filters'] = new_attr_dict[
                            'num_channels']
                        new_attr_dict['candidate_config'][
                            'channel'] = cur_channel
                    new_attr_dict['groups'] = new_attr_dict['num_channels']
                    layer = Block(
                        SuperDepthwiseConv2DTranspose(**new_attr_dict), key=key)
                else:
                    ### group conv_transpose
                    layer = Block(
                        SuperGroupConv2DTranspose(**new_attr_dict), key=key)
                model[idx] = layer

            elif isinstance(layer, Linear) and (
                    getattr(self.context, 'expand', None) != None or
                    getattr(self.context, 'channel', None) != None):
                attr_dict = layer.__dict__
                key = attr_dict['_full_name']
                ### TODO(paddle): add _param_attr and _bias_attr as private variable of Linear
                #new_attr_name = ['_act', '_dtype', '_param_attr', '_bias_attr']
                new_attr_name = ['_act', '_dtype']
                in_nc, out_nc = layer._parameters['weight'].shape

                new_attr_dict = dict()
                new_attr_dict['candidate_config'] = dict()
                if self.context.expand:
                    if idx == first_weight_layer_idx:
                        new_attr_dict['input_dim'] = int(in_nc)
                    else:
                        new_attr_dict['input_dim'] = self.context.expand * int(
                            in_nc)

                    if idx == last_weight_layer_idx:
                        new_attr_dict['output_dim'] = int(out_nc)
                    else:
                        new_attr_dict['output_dim'] = self.context.expand * int(
                            out_nc)
                        new_attr_dict['candidate_config'].update({
                            'expand_ratio': self.context.expand_ratio
                        })
                elif self.context.channel:
                    cur_channel = self.context.channel[0]
                    self.context.channel = self.context.channel[1:]
                    if idx == first_weight_layer_idx:
                        new_attr_dict['input_dim'] = int(in_nc)
                    else:
                        new_attr_dict['input_dim'] = max(pre_channel)

                    if idx == last_weight_layer_idx:
                        new_attr_dict['output_dim'] = int(out_nc)
                    else:
                        new_attr_dict['output_dim'] = max(cur_channel)
                        new_attr_dict['candidate_config'].update({
                            'channel': cur_channel
                        })
                        pre_channel = cur_channel
                else:
                    new_attr_dict['input_dim'] = int(in_nc)
                    new_attr_dict['output_dim'] = int(out_nc)

                for attr in new_attr_name:
                    new_attr_dict[attr[1:]] = attr_dict[attr]

                del layer, attr_dict

                layer = Block(SuperLinear(**new_attr_dict), key=key)
                model[idx] = layer

            elif isinstance(layer, InstanceNorm) and (
                    getattr(self.context, 'expand', None) != None or
                    getattr(self.context, 'channel', None) != None):
                # num_features in InstanceNorm don't change after last weight operators
                if idx > last_weight_layer_idx:
                    continue

                attr_dict = layer.__dict__
                new_attr_name = [
                    '_param_attr', '_bias_attr', '_dtype', '_epsilon'
                ]
                new_attr_dict = dict()
                if self.context.expand:
                    new_attr_dict['num_channels'] = self.context.expand * int(
                        layer._parameters['scale'].shape[0])
                elif self.context.channel:
                    new_attr_dict['num_channels'] = max(cur_channel)
                else:
                    new_attr_dict['num_channels'] = attr_dict['_num_channels']

                for attr in new_attr_name:
                    new_attr_dict[attr[1:]] = attr_dict[attr]

                del layer, attr_dict

                layer = SuperInstanceNorm(**new_attr_dict)
                model[idx] = layer

            elif isinstance(layer, LayerNorm) and (
                    getattr(self.context, 'expand', None) != None or
                    getattr(self.context, 'channel', None) != None):
                ### TODO(ceci3): fix when normalized_shape != last_dim_of_input
                if idx > last_weight_layer_idx:
                    continue

                attr_dict = layer.__dict__
                new_attr_name = [
                    '_scale', '_shift', '_param_attr', '_bias_attr', '_act',
                    '_dtype', '_epsilon'
                ]
                new_attr_dict = dict()
                if self.context.expand:
                    new_attr_dict[
                        'normalized_shape'] = self.context.expand * int(
                            attr_dict['_normalized_shape'][0])
                elif self.context.channel:
                    new_attr_dict['normalized_shape'] = max(cur_channel)
                else:
                    new_attr_dict['normalized_shape'] = attr_dict[
                        '_normalized_shape']

                for attr in new_attr_name:
                    new_attr_dict[attr[1:]] = attr_dict[attr]

                del layer, attr_dict
                layer = SuperLayerNorm(**new_attr_dict)
                model[idx] = layer

            elif isinstance(layer, Embedding) and (
                    getattr(self.context, 'expand', None) != None or
                    getattr(self.context, 'channel', None) != None):
                attr_dict = layer.__dict__
                key = attr_dict['_full_name']
                new_attr_name = [
                    '_is_sparse', '_is_distributed', '_padding_idx',
                    '_param_attr', '_dtype'
                ]

                new_attr_dict = dict()
                new_attr_dict['candidate_config'] = dict()
                bef_size = attr_dict['_size']
                if self.context.expand:
                    new_attr_dict['size'] = [
                        bef_size[0], self.context.expand * bef_size[1]
                    ]
                    new_attr_dict['candidate_config'].update({
                        'expand_ratio': self.context.expand_ratio
                    })

                elif self.context.channel:
                    cur_channel = self.context.channel[0]
                    self.context.channel = self.context.channel[1:]
                    new_attr_dict['size'] = [bef_size[0], max(cur_channel)]
                    new_attr_dict['candidate_config'].update({
                        'channel': cur_channel
                    })
                    pre_channel = cur_channel
                else:
                    new_attr_dict['size'] = bef_size

                for attr in new_attr_name:
                    new_attr_dict[attr[1:]] = attr_dict[attr]

                del layer, attr_dict

                layer = Block(SuperEmbedding(**new_attr_dict), key=key)
                model[idx] = layer

        return model


class supernet:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        assert (
            getattr(self, 'expand_ratio', None) == None or
            getattr(self, 'channel', None) == None
        ), "expand_ratio and channel CANNOT be NOT None at the same time."

        self.expand = None
        if 'expand_ratio' in kwargs.keys():
            if isinstance(self.expand_ratio, list) or isinstance(
                    self.expand_ratio, tuple):
                self.expand = max(self.expand_ratio)
            elif isinstance(self.expand_ratio, int):
                self.expand = self.expand_ratio

    def __enter__(self):
        return Convert(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


#def ofa_supernet(kernel_size, expand_ratio):
#    def _ofa_supernet(func):
#        @functools.wraps(func)
#        def convert(*args, **kwargs):
#            supernet_convert(*args, **kwargs)
#        return convert
#    return _ofa_supernet
