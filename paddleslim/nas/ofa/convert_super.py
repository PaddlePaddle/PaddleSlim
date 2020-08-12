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
import paddle
import paddle.nn as nn
import paddle.fluid as fluid
from paddle.fluid import framework
from paddleslim.core.layers import Block, SuperConv2D
from paddle.fluid.dygraph.nn import Conv2D
from ofa import OFA


### TODO: add decorator
class Convert:
    def __init__(self, context):
        self.context = context
        self.task = self.get_elastic_task

    def get_elastic_task(self):
        task = []
        if hasattr(self.context, 'kernel_size'):
            task += ['kernel']
        if hasattr(self.context, 'expand_stdio') or hasattr(
                self.context, 'channels') or hasattr(
                    self.context, 'embedding_size') or hasattr(
                        self.context, 'hidden_size') or hasattr(self.context,
                                                                'num_head'):
            task += ['width']
        if hasattr(self.context, 'depth'):
            task += ['depth']
        return task

    def convert(self, model, config=None):
        for idx, layer in enumerate(model):
            if isinstance(layer, nn.Conv2D):
                attr_dict = layer.__dict__
                key = attr_dict['_full_name']
                # TODO(lvmengsi): the channel of last conv and input conv donnot need to change
                # TODO(lvmengsi): if this conv is second conv in SeparableConv, don't change it.
                # TODO(lvmengsi): if pre conv is standard conv don't have in_channel_list.
                if attr_dict['_filter_size'] == '1':
                    continue

                new_attr_name = [
                    '_num_channels', '_stride', '_dilation', '_groups',
                    '_param_attr', '_bias_attr', '_use_cudnn', '_act', '_dtype'
                ]

                new_attr_dict = dict()
                new_attr_dict['candidate_config'] = dict()
                self.kernel_size = getattr(self.context, 'kernel_size', None)

                if self.kernel_size:
                    new_attr_dict['filter_size'] = max(self.kernel_size)
                    new_attr_dict['candidate_config'].update({
                        'kernel_size': self.kernel_size
                    })
                else:
                    new_attr_dict['filter_size'] = attr_dict['_filter_size']

                # TODO: make sure in_channels in 
                # https://github.com/mit-han-lab/once-for-all/blob/4decacf9d85dbc948a902c6dc34ea032d711e0a9/ofa/elastic_nn/modules/dynamic_layers.py#L31
                if self.context.expand:
                    new_attr_dict[
                        'num_filters'] = self.context.expand * attr_dict[
                            '_num_filters']
                    new_attr_dict['candidate_config'].update({
                        'num_filter': self.context.expand_stdio
                    })
                else:
                    new_attr_dict['num_filters'] = attr_dict['_num_filters']

                for attr in new_attr_name:
                    new_attr_dict[attr[1:]] = attr_dict[attr]
                del attr_dict, layer

                layer = Block(SuperConv2D(**new_attr_dict), key=key)
                model[idx] = layer

            elif isinstance(layer, nn.BatchNorm) and self.context.expand:
                attr_dict = layer.__dict__
                new_attr_name = [
                    '_param_attr', '_bias_attr', '_act', '_dtype', '_in_place',
                    '_data_layout', '_momentum', '_epsilon', '_is_test',
                    '_use_global_stats', '_trainable_statistics'
                ]
                new_attr_dict = dict()
                new_attr_dict['num_channels'] = self.context.expand * int(
                    layer._parameters['weight'].shape[0])
                for attr in new_attr_name:
                    new_attr_dict[attr[1:]] = attr_dict[attr]

                del layer, attr_dict
                ### TODO: change to SuperBatchNorm
                layer = nn.BatchNorm(**new_attr_dict)
                model[idx] = layer

            ### TODO: complete
            elif isinstance(layer, nn.Conv2DTranspose):
                pass

            ### TODO: complete
            elif isinstance(layer, nn.Linear):
                pass

            ### TODO: complete
            elif isinstance(layer, nn.InstanceNorm) and self.expand:
                pass

        return model


class ofa_supernet:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        #self.kernel_size = kernel_size
        #self.expand_stdio = expand_stdio

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
            models += [nn.Conv2D(3, 5, 3)]
            models += [nn.BatchNorm(5)]
            models += [nn.ReLU()]
            models = ofa_super.convert(models)
        self.models = nn.Sequential(*models)

    def forward(self, inputs):
        return self.models(inputs)


if __name__ == '__main__':
    import numpy as np
    data_np = np.random.random((1, 3, 10, 10)).astype('float32')
    fluid.enable_dygraph()
    net = Model()
    ofa_model = OFA(net)
    #print(net.__dict__)

    #for name, sublayer in net.named_sublayers():
    #    print(name, sublayer)

    data = fluid.dygraph.to_variable(data_np)
    output = ofa_model(data)
    print(output.numpy())
