#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV3_dy(fluid.dygraph.Layer):
    def __init__(self, scale=1.0, model_name='small', class_dim=1000):
        super(MobileNetV3_dy, self).__init__()

        inplanes = 16
        if model_name == "large":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hard_swish', 2],
                [3, 200, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],
                [5, 672, 160, True, 'hard_swish', 2],
                [5, 960, 160, True, 'hard_swish', 1],
                [5, 960, 160, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 960
            self.cls_ch_expand = 1280
        elif model_name == "small":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hard_swish', 2],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                [5, 288, 96, True, 'hard_swish', 2],
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 576
            self.cls_ch_expand = 1280
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        # network
        # conv1
        self.conv1 = ConvBnLayer(
            in_c=3,
            out_c=make_divisible(inplanes * scale),
            filter_size=3,
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act='hard_swish',
            name=self.full_name() + 'conv1')
        # conv_blocks
        self.all_blocks = []
        i = 0
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s) in self.cfg:
            self.all_blocks.append(
                ResidualUnit(
                    in_c=inplanes,
                    mid_c=make_divisible(scale * exp),
                    out_c=make_divisible(scale * c),
                    filter_size=k,
                    stride=s,
                    ues_se=se,
                    act=nl,
                    name=self.full_name() + 'conv' + str(i + 2)))
            self.add_sublayer(
                sublayer=self.all_blocks[-1],
                name=self.full_name() + 'conv' + str(i + 2))
            inplanes = make_divisible(scale * c)
            i += 1
        # last_second_conv
        self.last_second_conv = ConvBnLayer(
            in_c=inplanes,
            out_c=make_divisible(scale * self.cls_ch_squeeze),
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            act='hard_swish',
            name=self.full_name() + 'last_second_conv')
        # global_avg_pool
        self.pool = fluid.dygraph.Pool2D(
            pool_type='avg', global_pooling=True, use_cudnn=False)
        # last_conv
        self.last_conv = fluid.dygraph.Conv2D(
            num_channels=make_divisible(scale * self.cls_ch_squeeze),
            num_filters=self.cls_ch_expand,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(
                name=self.full_name() + 'last_1x1_conv_weights'),
            bias_attr=False)
        # fc
        self.fc = fluid.dygraph.Linear(
            input_dim=self.cls_ch_expand,
            output_dim=class_dim,
            param_attr=ParamAttr(name=self.full_name() + 'fc_weights'))

    def forward(self, inputs, label=None, dropout_prob=0.2):
        x = self.conv1(inputs)
        for i in range(len(self.all_blocks)):
            x = self.all_blocks[i](x)
        x = self.last_second_conv(x)
        x = self.pool(x)
        x = self.last_conv(x)
        x = fluid.layers.hard_swish(x)
        x = fluid.layers.dropout(x=x, dropout_prob=dropout_prob)
        x = fluid.layers.reshape(x, shape=[x.shape[0], x.shape[1]])
        x = self.fc(x)

        if label:
            acc1 = fluid.layers.accuracy(input=x, label=label)
            acc5 = fluid.layers.accuracy(input=x, label=label, k=5)
            return x, acc1, acc5
        return x


class ConvBnLayer(fluid.dygraph.Layer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1,
                 if_act=True,
                 act=None,
                 use_cudnn=True,
                 name=''):
        super(ConvBnLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = fluid.dygraph.Conv2D(
            num_channels=in_c,
            num_filters=out_c,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            use_cudnn=use_cudnn,
            act=None)
        self.bn = fluid.dygraph.BatchNorm(
            num_channels=out_c,
            act=None,
            param_attr=ParamAttr(
                name=name + "_bn" + "_scale",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0)),
            bias_attr=ParamAttr(
                name=name + "_bn" + "_offset",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0)),
            moving_mean_name=name + "_bn" + '_mean',
            moving_variance_name=name + "_bn" + '_variance')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == 'relu':
                x = fluid.layers.relu(x)
            elif self.act == 'hard_swish':
                x = fluid.layers.hard_swish(x)
            else:
                print('The activation function is selected incorrectly.')
                exit()
        return x


class ResidualUnit(fluid.dygraph.Layer):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 ues_se,
                 act=None,
                 name=''):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.if_se = ues_se

        self.expand_conv = ConvBnLayer(
            in_c=in_c,
            out_c=mid_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
            name=name + '_expand')
        self.bottleneck_conv = ConvBnLayer(
            in_c=mid_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            num_groups=mid_c,
            if_act=True,
            act=act,
            name=name + '_depthwise')
        if self.if_se:
            self.mid_se = SEModule(mid_c, name=name + '_SE')
        self.linear_conv = ConvBnLayer(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None,
            name=name + '_linear')

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = fluid.layers.elementwise_add(inputs, x)
        return x


class SEModule(fluid.dygraph.Layer):
    def __init__(self, channel, reduction=4, name=''):
        super(SEModule, self).__init__()
        self.avg_pool = fluid.dygraph.Pool2D(
            pool_type='avg', global_pooling=True, use_cudnn=False)
        self.conv1 = fluid.dygraph.Conv2D(
            num_channels=channel,
            num_filters=channel // reduction,
            filter_size=1,
            stride=1,
            padding=0,
            act='relu',
            param_attr=ParamAttr(name=name + "_weights1"),
            bias_attr=ParamAttr(name=name + "_bias1"))
        self.conv2 = fluid.dygraph.Conv2D(
            num_channels=channel // reduction,
            num_filters=channel,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(name=name + "_weights2"),
            bias_attr=ParamAttr(name=name + "_bias2"))

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = fluid.layers.hard_sigmoid(outputs, slope=0.2, offset=0.5)
        return fluid.layers.elementwise_mul(x=inputs, y=outputs, axis=0)


if __name__ == "__main__":
    import numpy as np
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = MobileNetV3_dy(scale=0.6, model_name='large', class_dim=1000)

        img = np.random.uniform(0, 255, [8, 3, 224, 224]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        res = model(img)
        print(res.shape)
    #     out_dygraph, static_layer = fluid.dygraph.TracedLayer.trace(model, inputs=[img])

    #     out_static_graph = static_layer([img])
    #     print(len(out_static_graph)) # 1
    #     print(out_static_graph[0].shape) 

    #     static_layer.save_inference_model(dirname='./saved_infer_model')

    # from calc_flops import *
    # place = fluid.CPUPlace()
    # exe = fluid.Executor(place)
    # program, feed_vars, fetch_vars = fluid.io.load_inference_model('./saved_infer_model', exe)

    # total_flops_params, is_quantize = summary(program)
