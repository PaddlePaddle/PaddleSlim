import paddle
from paddle.nn.initializer import KaimingUniform
import math

__all__ = [
    'MobileNetV3', 'MobileNetV3_small_x0_25', 'MobileNetV3_small_x0_5',
    'MobileNetV3_small_x0_75', 'MobileNetV3_small_x1_0',
    'MobileNetV3_small_x1_25', 'MobileNetV3_large_x0_25',
    'MobileNetV3_large_x0_5', 'MobileNetV3_large_x0_75',
    'MobileNetV3_large_x1_0', 'MobileNetV3_large_x1_25',
    'MobileNetV3_large_x2_0'
]


class MobileNetV3():
    def __init__(self, scale=1.0, model_name='small'):
        self.scale = scale
        self.inplanes = 16
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
            raise NotImplementedError

    def net(self, input, class_dim=1000):
        scale = self.scale
        inplanes = self.inplanes
        cfg = self.cfg
        cls_ch_squeeze = self.cls_ch_squeeze
        cls_ch_expand = self.cls_ch_expand

        #conv1
        conv = self.conv_bn_layer(
            input,
            filter_size=3,
            #num_filters=int(scale*inplanes),
            num_filters=inplanes if scale <= 1.0 else int(inplanes * scale),
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act='hard_swish',
            name='conv1')
        i = 0
        for layer_cfg in cfg:
            conv = self.residual_unit(
                input=conv,
                num_in_filter=inplanes,
                num_mid_filter=int(scale * layer_cfg[1]),
                num_out_filter=int(scale * layer_cfg[2]),
                act=layer_cfg[4],
                stride=layer_cfg[5],
                filter_size=layer_cfg[0],
                use_se=layer_cfg[3],
                name='conv' + str(i + 2))

            inplanes = int(scale * layer_cfg[2])
            i += 1

        conv = self.conv_bn_layer(
            input=conv,
            filter_size=1,
            num_filters=int(scale * cls_ch_squeeze),
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            act='hard_swish',
            name='conv_last')
        conv = paddle.nn.functional.adaptive_avg_pool2d(conv, 1)
        conv = paddle.static.nn.conv2d(
            input=conv,
            num_filters=cls_ch_expand,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=paddle.ParamAttr(name='last_1x1_conv_weights'),
            bias_attr=False)
        conv = paddle.nn.functional.hardswish(conv)
        out = paddle.static.nn.fc(
            conv,
            class_dim,
            weight_attr=paddle.ParamAttr(name='fc_weights'),
            bias_attr=paddle.ParamAttr(name='fc_offset'))
        return out

    def conv_bn_layer(self,
                      input,
                      filter_size,
                      num_filters,
                      stride,
                      padding,
                      num_groups=1,
                      if_act=True,
                      act=None,
                      name=None,
                      use_cudnn=True):
        conv = paddle.static.nn.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=paddle.ParamAttr(name=name + '_weights'),
            bias_attr=False)
        bn_name = name + '_bn'
        bn = paddle.static.nn.batch_norm(
            input=conv,
            param_attr=paddle.ParamAttr(
                name=bn_name + "_scale",
                regularizer=paddle.regularizer.L2Decay(coeff=0.0)),
            bias_attr=paddle.ParamAttr(
                name=bn_name + "_offset",
                regularizer=paddle.regularizer.L2Decay(coeff=0.0)),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')
        if if_act:
            if act == 'relu':
                bn = paddle.nn.functional.relu(bn)
            elif act == 'hard_swish':
                bn = paddle.nn.functional.hardswish(bn)
        return bn

    def hard_swish(self, x):
        return x * paddle.nn.functional.relu6(x + 3) / 6.

    def se_block(self, input, num_out_filter, ratio=4, name=None):
        num_mid_filter = int(num_out_filter // ratio)
        pool = paddle.nn.functional.adaptive_avg_pool2d(input, 1)
        conv1 = paddle.static.nn.conv2d(
            input=pool,
            filter_size=1,
            num_filters=num_mid_filter,
            act='relu',
            param_attr=paddle.ParamAttr(name=name + '_1_weights'),
            bias_attr=paddle.ParamAttr(name=name + '_1_offset'))
        conv2 = paddle.static.nn.conv2d(
            input=conv1,
            filter_size=1,
            num_filters=num_out_filter,
            act='hard_sigmoid',
            param_attr=paddle.ParamAttr(name=name + '_2_weights'),
            bias_attr=paddle.ParamAttr(name=name + '_2_offset'))

        scale = paddle.multiply(x=input, y=conv2)
        return scale

    def residual_unit(self,
                      input,
                      num_in_filter,
                      num_mid_filter,
                      num_out_filter,
                      stride,
                      filter_size,
                      act=None,
                      use_se=False,
                      name=None):

        input_data = input
        conv0 = self.conv_bn_layer(
            input=input,
            filter_size=1,
            num_filters=num_mid_filter,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
            name=name + '_expand')

        conv1 = self.conv_bn_layer(
            input=conv0,
            filter_size=filter_size,
            num_filters=num_mid_filter,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            if_act=True,
            act=act,
            num_groups=num_mid_filter,
            use_cudnn=False,
            name=name + '_depthwise')

        if use_se:
            with paddle.static.name_scope('se_block_skip'):
                conv1 = self.se_block(
                    input=conv1,
                    num_out_filter=num_mid_filter,
                    name=name + '_se')

        conv2 = self.conv_bn_layer(
            input=conv1,
            filter_size=1,
            num_filters=num_out_filter,
            stride=1,
            padding=0,
            if_act=False,
            name=name + '_linear')
        if num_in_filter != num_out_filter or stride != 1:
            return conv2
        else:
            return paddle.add(x=input_data, y=conv2)


def MobileNetV3_small_x0_25():
    model = MobileNetV3(model_name='small', scale=0.25)
    return model


def MobileNetV3_small_x0_5():
    model = MobileNetV3(model_name='small', scale=0.5)
    return model


def MobileNetV3_small_x0_75():
    model = MobileNetV3(model_name='small', scale=0.75)
    return model


def MobileNetV3_small_x1_0():
    model = MobileNetV3(model_name='small', scale=1.0)
    return model


def MobileNetV3_small_x1_25():
    model = MobileNetV3(model_name='small', scale=1.25)
    return model


def MobileNetV3_large_x0_25():
    model = MobileNetV3(model_name='large', scale=0.25)
    return model


def MobileNetV3_large_x0_5():
    model = MobileNetV3(model_name='large', scale=0.5)
    return model


def MobileNetV3_large_x0_75():
    model = MobileNetV3(model_name='large', scale=0.75)
    return model


def MobileNetV3_large_x1_0():
    model = MobileNetV3(model_name='large', scale=1.0)
    return model


def MobileNetV3_large_x1_25():
    model = MobileNetV3(model_name='large', scale=1.25)
    return model


def MobileNetV3_large_x2_0():
    model = MobileNetV3(model_name='large', scale=2.0)
    return model
