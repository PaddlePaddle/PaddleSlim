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
import paddle.nn as nn
import paddle.fluid as fluid
from paddle.fluid import framework
from paddleslim.core.layers import *
from paddle.fluid.dygraph.nn import Conv2D
from ofa import OFA

WEIGHT_LAYER = ['conv', 'linear']


### TODO: add decorator
class Convert:
    def __init__(self, context):
        self.context = context
        self.task = self.get_elastic_task

    def get_elastic_task(self):
        task = []
        if hasattr(self.context, 'kernel_size'):
            task += ['kernel']

    # or hasattr(self.context, 'embedding_size') or hasattr(self.context, 'hidden_size') or hasattr(self.context, 'num_head'):
        if hasattr(self.context, 'expand_stdio') or hasattr(self.context,
                                                            'channel'):
            task += ['width']
        if hasattr(self.context, 'depth'):
            task += ['depth']
        return task

    def convert(self, model, config=None):
        # search the first and last weight layer, don't change out channel of the last weight layer
        # don't change in channel of the first weight layer
        first_weight_layer_idx = -1
        last_weight_layer_idx = -1
        for idx, layer in enumerate(model):
            cls_name = layer.__class__.__name__.lower()
            if 'conv' in cls_name or 'linear' in cls_name:
                last_weight_layer_idx = idx
                if first_weight_layer_idx == -1:
                    first_weight_layer_idx = idx

        for idx, layer in enumerate(model):
            if isinstance(layer, nn.Conv2D):
                attr_dict = layer.__dict__
                key = attr_dict['_full_name']

                new_attr_name = [
                    '_stride', '_dilation', '_groups', '_param_attr',
                    '_bias_attr', '_use_cudnn', '_act', '_dtype'
                ]

                new_attr_dict = dict()
                new_attr_dict['candidate_config'] = dict()
                self.kernel_size = getattr(self.context, 'kernel_size', None)

                # if the kernel_size of conv is 1, don't change it.
                if self.kernel_size and int(attr_dict['_filter_size'][0]) != 1:
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
                            'expand_stdio': self.context.expand_stdio
                        })
                elif self.context.channel:
                    if idx == first_weight_layer_idx:
                        new_attr_dict['num_channels'] = attr_dict[
                            '_num_channels']
                    else:
                        new_attr_dict['num_channels'] = max(
                            self.context.channel)

                    if idx == last_weight_layer_idx:
                        new_attr_dict['num_filters'] = attr_dict['_num_filters']
                    else:
                        new_attr_dict['num_filters'] = max(self.context.channel)
                        new_attr_dict['candidate_config'].update({
                            'channel': self.context.channel
                        })
                else:
                    new_attr_dict['num_filters'] = attr_dict['_num_filters']
                    new_attr_dict['num_channels'] = attr_dict['_num_channels']

                for attr in new_attr_name:
                    new_attr_dict[attr[1:]] = attr_dict[attr]

                del layer

                if int(attr_dict['_groups']) == 1:
                    ### standard conv
                    layer = Block(SuperConv2D(**new_attr_dict), key=key)
                elif int(attr_dict['_groups']) == int(attr_dict[
                        '_num_filters']):
                    ### depthwise conv
                    logging.warning(
                        "If convolution is a depthwise conv, output channel change to the same channel with input, output channel in search is not used."
                    )
                    new_attr_dict['groups'] = new_attr_dict['num_filters']
                    layer = Block(
                        SuperDepthwiseConv2D(**new_attr_dict), key=key)
                else:
                    ### group conv
                    layer = Block(SuperGroupConv2D(**new_attr_dict), key=key)
                model[idx] = layer

            elif isinstance(layer, nn.BatchNorm) and (
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
                    new_attr_dict['num_channels'] = max(self.context.channel)

                for attr in new_attr_name:
                    new_attr_dict[attr[1:]] = attr_dict[attr]

                del layer, attr_dict

                layer = SuperBatchNorm(**new_attr_dict)
                model[idx] = layer

            ### assume output_size = None, filter_size != None
            ### output_size != None may raise error, solve when it happend. 
            elif isinstance(layer, nn.Conv2DTranspose):
                attr_dict = layer.__dict__
                key = attr_dict['_full_name']

                new_attr_name = [
                    '_stride', '_dilation', '_groups', '_param_attr',
                    '_bias_attr', '_use_cudnn', '_act', '_dtype', '_output_size'
                ]
                assert attr_dict[
                    '_filter_size'] != None, "Conv2DTranspose only support filter size != None now"

                new_attr_dict = dict()
                new_attr_dict['candidate_config'] = dict()
                self.kernel_size = getattr(self.context, 'kernel_size', None)

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
                            'expand_stdio': self.context.expand_stdio
                        })
                elif self.context.channel:
                    if idx == first_weight_layer_idx:
                        new_attr_dict['num_channels'] = attr_dict[
                            '_num_channels']
                    else:
                        new_attr_dict['num_channels'] = max(
                            self.context.channel)

                    if idx == last_weight_layer_idx:
                        new_attr_dict['num_filters'] = attr_dict['_num_filters']
                    else:
                        new_attr_dict['num_filters'] = max(self.context.channel)
                        new_attr_dict['candidate_config'].update({
                            'channel': self.context.channel
                        })
                else:
                    new_attr_dict['num_filters'] = attr_dict['_num_filters']
                    new_attr_dict['num_channels'] = attr_dict['_num_channels']

                for attr in new_attr_name:
                    new_attr_dict[attr[1:]] = attr_dict[attr]

                del layer

                if int(attr_dict['_groups']) == 1:
                    ### standard conv_transpose
                    layer = Block(
                        SuperConv2DTranspose(**new_attr_dict), key=key)
                elif int(attr_dict['_groups']) == int(attr_dict[
                        '_num_filters']):
                    ### depthwise conv_transpose
                    logging.warning(
                        "If convolution is a depthwise conv_transpose, output channel change to the same channel with input, output channel in search is not used."
                    )
                    new_attr_dict['groups'] = new_attr_dict['num_filters']
                    layer = Block(
                        SuperDepthwiseConv2DTranspose(**new_attr_dict), key=key)
                else:
                    ### group conv_transpose
                    layer = Block(
                        SuperGroupConv2DTranspose(**new_attr_dict), key=key)
                model[idx] = layer

            ### TODO(paddle): add _param_attr and _bias_attr as private variable of Linear
            elif isinstance(layer, nn.Linear) and (
                    getattr(self.context, 'expand', None) != None or
                    getattr(self.context, 'channel', None) != None):
                attr_dict = layer.__dict__
                key = attr_dict['_full_name']
                new_attr_name = ['_act', '_dtype', '_param_attr', '_bias_attr']
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
                            'expand_stdio': self.context.expand_stdio
                        })
                elif self.context.channel:
                    if idx == first_weight_layer_idx:
                        new_attr_dict['input_dim'] = int(in_nc)
                    else:
                        new_attr_dict['input_dim'] = max(self.context.channel)

                    if idx == last_weight_layer_idx:
                        new_attr_dict['output_dim'] = int(out_nc)
                    else:
                        new_attr_dict['output_dim'] = max(self.context.channel)
                        new_attr_dict['candidate_config'].update({
                            'channel': self.context.channel
                        })
                else:
                    new_attr_dict['input_dim'] = int(in_nc)
                    new_attr_dict['output_dim'] = int(out_nc)

                for attr in new_attr_name:
                    new_attr_dict[attr[1:]] = attr_dict[attr]

                del layer, attr_dict

                layer = Block(SuperLinear(**new_attr_dict), key=key)
                model[idx] = layer

            elif isinstance(layer, nn.InstanceNorm) and (
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
                    new_attr_dict['num_channels'] = max(self.context.channel)

                for attr in new_attr_name:
                    new_attr_dict[attr[1:]] = attr_dict[attr]

                del layer, attr_dict

                layer = SuperInstanceNorm(**new_attr_dict)
                model[idx] = layer

        return model


class ofa_supernet:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        assert (
            getattr(self, 'expand_stdio', None) == None or
            getattr(self, 'channel', None) == None
        ), "expand_stdio and channel CANNOT be NOT None at the same time."

        self.expand = None
        if 'expand_stdio' in kwargs.keys():
            if isinstance(self.expand_stdio, list) or isinstance(
                    self.expand_stdio, tuple):
                self.expand = max(self.expand_stdio)
            elif isinstance(self.expand_stdio, int):
                self.expand = self.expand_stdio

    def __enter__(self):
        return Convert(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


#def ofa_supernet(kernel_size, expand_stdio):
#    def _ofa_supernet(func):
#        @functools.wraps(func)
#        def convert(*args, **kwargs):
#            supernet_convert(*args, **kwargs)
#        return convert
#    return _ofa_supernet


class Model(fluid.dygraph.Layer):
    #@ofa_supernet(kernel_size=(3,5,7), expand_stdio=(1, 2, 4))
    def __init__(self):
        super(Model, self).__init__()
        with ofa_supernet(
                kernel_size=(3, 5, 7), expand_stdio=(1, 2, 4)) as ofa_super:
            models = []
            models += [nn.Conv2D(3, 4, 3)]
            models += [nn.InstanceNorm(4)]
            models += [nn.ReLU()]
            models += [nn.Conv2DTranspose(4, 4, 3, groups=4, use_cudnn=False)]
            models += [nn.BatchNorm(4)]
            models += [nn.ReLU()]
            models += [
                fluid.dygraph.Pool2D(
                    pool_type='avg', global_pooling=True, use_cudnn=False)
            ]
            models += [nn.Linear(4, 3)]
            models += [nn.ReLU()]
            models = ofa_super.convert(models)
        self.models = nn.Sequential(*models)

    def forward(self, inputs):
        for idx, layer in enumerate(self.models):
            if idx == (len(self.models) - 2):
                inputs = fluid.layers.reshape(
                    inputs, shape=[inputs.shape[0], -1])
            inputs = layer(inputs)
        return inputs
        #return self.models(inputs)


if __name__ == '__main__':
    import numpy as np
    data_np = np.random.random((1, 3, 10, 10)).astype('float32')
    fluid.enable_dygraph()
    net = Model()
    ofa_model = OFA(net)
    adam = fluid.optimizer.Adam(
        learning_rate=0.001, parameter_list=ofa_model.parameters())

    for name, sublayer in net.named_sublayers():
        print(name, sublayer)
        if getattr(sublayer, '_filter_size', None) != None and getattr(
                sublayer, '_num_filters', None) != None:
            print(name, sublayer._num_channels, sublayer._num_filters,
                  sublayer._filter_size)
        if getattr(sublayer, 'candidate_config', None) != None:
            print(name, sublayer.candidate_config)

    data = fluid.dygraph.to_variable(data_np)
    for _ in range(10):
        out = ofa_model(data)
        loss = fluid.layers.reduce_mean(out)
        print(loss.numpy())
        loss.backward()
        adam.minimize(loss)
        adam.clear_gradients()
