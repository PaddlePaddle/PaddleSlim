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
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import InstanceNorm, Conv2D, Conv2DTranspose, BatchNorm
from paddle.nn.layer import Leaky_ReLU, ReLU, Pad2D


class NLayerDiscriminator(fluid.dygraph.Layer):
    def __init__(self, input_channel, ndf, n_layers=3,
                 norm_layer=InstanceNorm):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == InstanceNorm
        else:
            use_bias = norm_layer == InstanceNorm

        kw = 4
        padw = 1
        self.model = fluid.dygraph.LayerList([
            Conv2D(
                input_channel, ndf, filter_size=kw, stride=2, padding=padw),
            Leaky_ReLU(0.2)
        ])
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            self.model.extend([
                Conv2D(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    filter_size=kw,
                    stride=2,
                    padding=padw,
                    bias_attr=use_bias),
                #norm_layer(ndf * nf_mult),
                InstanceNorm(
                    ndf * nf_mult,
                    param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Constant(1.0),
                        learning_rate=0.0,
                        trainable=False),
                    bias_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Constant(0.0),
                        learning_rate=0.0,
                        trainable=False)),
                Leaky_ReLU(0.2)
            ])

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        self.model.extend([
            Conv2D(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                filter_size=kw,
                stride=1,
                padding=padw,
                bias_attr=use_bias),
            #norm_layer(ndf * nf_mult),
            InstanceNorm(
                ndf * nf_mult,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(1.0),
                    learning_rate=0.0,
                    trainable=False),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(0.0),
                    learning_rate=0.0,
                    trainable=False)),
            Leaky_ReLU(0.2)
        ])

        self.model.extend([
            Conv2D(
                ndf * nf_mult, 1, filter_size=kw, stride=1, padding=padw)
        ])

    def forward(self, inputs):
        #import numpy as np
        #print("================ DISCRIMINATOR ====================")
        y = inputs
        for sublayer in self.model:
            y = sublayer(y)
        #    print(sublayer, np.sum(np.abs(y.numpy())))
        #print("===================================================")
        return y
