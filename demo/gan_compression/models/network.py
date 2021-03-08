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
import functools
import paddle
import paddle.nn as nn
from paddle.nn import BatchNorm2D, InstanceNorm2D
from .discrimitor import NLayerDiscriminator
from .generator.resnet_generator import ResnetGenerator
from .generator.mobile_generator import MobileResnetGenerator
from .generator.super_generator import SuperMobileResnetGenerator
from .generator.sub_mobile_generator import SubMobileResnetGenerator


class Identity(nn.Layer):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'instance':
        norm_layer = functools.partial(
            InstanceNorm2D, weight_attr=False, bias_attr=False)
    elif norm_type == 'batch':
        norm_layer = functools.partial(
            BatchNorm2D,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(1.0, 0.02)),
            bias_attr=paddle.ParamAttr(
                initializer=nn.initializer.Constant(0.0)))
    elif norm_type == 'none':

        def norm_layer(x):
            return Identity(x)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' %
                                  norm_type)
    return norm_layer


def define_G(input_nc,
             output_nc,
             ngf,
             netG,
             norm_type='batch',
             dropout_rate=0,
             init_type='normal',
             stddev=0.02):
    net = None
    norm_layer = get_norm_layer(norm_type)
    if netG == 'resnet_9blocks':
        net = ResnetGenerator(
            input_nc,
            output_nc,
            ngf,
            norm_layer=norm_layer,
            dropout_rate=dropout_rate,
            n_blocks=9)
    elif netG == 'mobile_resnet_9blocks':
        net = MobileResnetGenerator(
            input_nc,
            output_nc,
            ngf,
            norm_layer=norm_layer,
            dropout_rate=dropout_rate,
            n_blocks=9)
    elif netG == 'super_mobile_resnet_9blocks':
        net = SuperMobileResnetGenerator(
            input_nc,
            output_nc,
            ngf,
            norm_layer=norm_layer,
            dropout_rate=dropout_rate,
            n_blocks=9)
    elif netG == 'sub_mobile_resnet_9blocks':
        assert self.cfgs.config_str is not None
        config = decode_config(self.cfgs.config_str)
        net = SubMobileResnetGenerator(
            input_nc,
            output_nc,
            config,
            norm_layer=norm_layer,
            dropout_rate=dropout_rate,
            n_blocks=9)
    return net


def define_D(input_nc, ndf, netD, norm_type='batch', n_layers_D=3):
    net = None
    norm_layer = get_norm_layer(norm_type)
    if netD == 'n_layers':
        net = NLayerDiscriminator(
            input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    return net
