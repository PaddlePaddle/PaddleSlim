from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr

__all__ = [
    'MobileNet', 'MobileNetSensitive30', 'MobileNetSensitive50',
    'MobileNetCifar'
]

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [10, 16, 30],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}


class MobileNet():
    def __init__(self):
        self.params = train_parameters

    def net(self, input, class_dim=1000, scale=1.0):
        # conv1: 112x112
        input = self.conv_bn_layer(
            input,
            filter_size=3,
            channels=3,
            num_filters=int(32 * scale),
            stride=2,
            padding=1,
            name="conv1")

        # 56x56
        input = self.depthwise_separable(
            input,
            num_filters1=32,
            num_filters2=64,
            num_groups=32,
            stride=1,
            scale=scale,
            name="conv2_1")

        input = self.depthwise_separable(
            input,
            num_filters1=64,
            num_filters2=128,
            num_groups=64,
            stride=2,
            scale=scale,
            name="conv2_2")

        # 28x28
        input = self.depthwise_separable(
            input,
            num_filters1=128,
            num_filters2=128,
            num_groups=128,
            stride=1,
            scale=scale,
            name="conv3_1")

        input = self.depthwise_separable(
            input,
            num_filters1=128,
            num_filters2=256,
            num_groups=128,
            stride=2,
            scale=scale,
            name="conv3_2")

        # 14x14
        input = self.depthwise_separable(
            input,
            num_filters1=256,
            num_filters2=256,
            num_groups=256,
            stride=1,
            scale=scale,
            name="conv4_1")

        input = self.depthwise_separable(
            input,
            num_filters1=256,
            num_filters2=512,
            num_groups=256,
            stride=2,
            scale=scale,
            name="conv4_2")

        # 14x14
        for i in range(5):
            input = self.depthwise_separable(
                input,
                num_filters1=512,
                num_filters2=512,
                num_groups=512,
                stride=1,
                scale=scale,
                name="conv5" + "_" + str(i + 1))
        # 7x7
        input = self.depthwise_separable(
            input,
            num_filters1=512,
            num_filters2=1024,
            num_groups=512,
            stride=2,
            scale=scale,
            name="conv5_6")

        input = self.depthwise_separable(
            input,
            num_filters1=1024,
            num_filters2=1024,
            num_groups=1024,
            stride=1,
            scale=scale,
            name="conv6")

        input = fluid.layers.pool2d(
            input=input,
            pool_size=0,
            pool_stride=1,
            pool_type='avg',
            global_pooling=True)
        with fluid.name_scope('last_fc'):
            output = fluid.layers.fc(input=input,
                                     size=class_dim,
                                     param_attr=ParamAttr(
                                         initializer=MSRA(),
                                         name="fc7_weights"),
                                     bias_attr=ParamAttr(name="fc7_offset"))

        return output

    def conv_bn_layer(self,
                      input,
                      filter_size,
                      num_filters,
                      stride,
                      padding,
                      channels=None,
                      num_groups=1,
                      act='relu',
                      use_cudnn=True,
                      name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(
                initializer=MSRA(), name=name + "_weights"),
            bias_attr=False)
        bn_name = name + "_bn"
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def depthwise_separable(self,
                            input,
                            num_filters1,
                            num_filters2,
                            num_groups,
                            stride,
                            scale,
                            name=None):
        depthwise_conv = self.conv_bn_layer(
            input=input,
            filter_size=3,
            num_filters=int(num_filters1 * scale),
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=False,
            name=name + "_dw")

        pointwise_conv = self.conv_bn_layer(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0,
            name=name + "_sep")
        return pointwise_conv


class MobileNetCifar():
    def __init__(self):
        self.params = train_parameters

    def net(self, input, class_dim=100, scale=1.0):
        # conv1: 32x32
        input = self.conv_bn_layer(
            input,
            filter_size=3,
            channels=3,
            num_filters=int(32 * scale),
            stride=1,
            padding=1,
            name="conv1")

        # 16x16
        input = self.depthwise_separable(
            input,
            num_filters1=32,
            num_filters2=64,
            num_groups=32,
            stride=1,
            scale=scale,
            name="conv2_1")

        input = self.depthwise_separable(
            input,
            num_filters1=64,
            num_filters2=128,
            num_groups=64,
            stride=2,
            scale=scale,
            name="conv2_2")

        # 8x8
        input = self.depthwise_separable(
            input,
            num_filters1=128,
            num_filters2=128,
            num_groups=128,
            stride=1,
            scale=scale,
            name="conv3_1")

        input = self.depthwise_separable(
            input,
            num_filters1=128,
            num_filters2=256,
            num_groups=128,
            stride=2,
            scale=scale,
            name="conv3_2")

        # 4x4
        input = self.depthwise_separable(
            input,
            num_filters1=256,
            num_filters2=256,
            num_groups=256,
            stride=1,
            scale=scale,
            name="conv4_1")

        input = self.depthwise_separable(
            input,
            num_filters1=256,
            num_filters2=512,
            num_groups=256,
            stride=2,
            scale=scale,
            name="conv4_2")

        # 4x4
        for i in range(5):
            input = self.depthwise_separable(
                input,
                num_filters1=512,
                num_filters2=512,
                num_groups=512,
                stride=1,
                scale=scale,
                name="conv5" + "_" + str(i + 1))
        # 4x4
        input = self.depthwise_separable(
            input,
            num_filters1=512,
            num_filters2=1024,
            num_groups=512,
            stride=1,
            scale=scale,
            name="conv5_6")

        input = self.depthwise_separable(
            input,
            num_filters1=1024,
            num_filters2=1024,
            num_groups=1024,
            stride=1,
            scale=scale,
            name="conv6")

        input = fluid.layers.pool2d(
            input=input,
            pool_size=0,
            pool_stride=1,
            pool_type='avg',
            global_pooling=True)

        with fluid.name_scope('last_fc'):
            output = fluid.layers.fc(input=input,
                                     size=class_dim,
                                     param_attr=ParamAttr(
                                         initializer=MSRA(),
                                         name="fc7_weights"),
                                     bias_attr=ParamAttr(name="fc7_offset"))

        return output

    def conv_bn_layer(self,
                      input,
                      filter_size,
                      num_filters,
                      stride,
                      padding,
                      channels=None,
                      num_groups=1,
                      act='relu',
                      use_cudnn=True,
                      name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(
                initializer=MSRA(), name=name + "_weights"),
            bias_attr=False)
        bn_name = name + "_bn"
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def depthwise_separable(self,
                            input,
                            num_filters1,
                            num_filters2,
                            num_groups,
                            stride,
                            scale,
                            name=None):
        depthwise_conv = self.conv_bn_layer(
            input=input,
            filter_size=3,
            num_filters=int(num_filters1 * scale),
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=False,
            name=name + "_dw")

        pointwise_conv = self.conv_bn_layer(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0,
            name=name + "_sep")
        return pointwise_conv


class MobileNetSensitive30(MobileNet):
    def net(self, input, class_dim=1000, scale=1.0):
        # conv1: 112x112
        input = self.conv_bn_layer(
            input,
            filter_size=3,
            channels=3,
            num_filters=int(32 * scale),
            stride=2,
            padding=1,
            name="conv1")

        # 56x56
        input = self.depthwise_separable(
            input,
            num_filters1=32,
            num_filters2=45,
            num_groups=32,
            stride=1,
            scale=scale,
            name="conv2_1")

        input = self.depthwise_separable(
            input,
            num_filters1=45,
            num_filters2=64,
            num_groups=45,
            stride=2,
            scale=scale,
            name="conv2_2")

        # 28x28
        input = self.depthwise_separable(
            input,
            num_filters1=64,
            num_filters2=122,
            num_groups=64,
            stride=1,
            scale=scale,
            name="conv3_1")

        input = self.depthwise_separable(
            input,
            num_filters1=122,
            num_filters2=186,
            num_groups=122,
            stride=2,
            scale=scale,
            name="conv3_2")

        # 14x14
        input = self.depthwise_separable(
            input,
            num_filters1=186,
            num_filters2=231,
            num_groups=186,
            stride=1,
            scale=scale,
            name="conv4_1")

        input = self.depthwise_separable(
            input,
            num_filters1=231,
            num_filters2=383,
            num_groups=231,
            stride=2,
            scale=scale,
            name="conv4_2")

        # 14x14
        input = self.depthwise_separable(
            input,
            num_filters1=383,
            num_filters2=491,
            num_groups=383,
            stride=1,
            scale=scale,
            name="conv5" + "_" + str(1))

        input = self.depthwise_separable(
            input,
            num_filters1=491,
            num_filters2=492,
            num_groups=491,
            stride=1,
            scale=scale,
            name="conv5" + "_" + str(2))

        input = self.depthwise_separable(
            input,
            num_filters1=492,
            num_filters2=471,
            num_groups=492,
            stride=1,
            scale=scale,
            name="conv5" + "_" + str(3))

        input = self.depthwise_separable(
            input,
            num_filters1=471,
            num_filters2=459,
            num_groups=471,
            stride=1,
            scale=scale,
            name="conv5" + "_" + str(4))

        input = self.depthwise_separable(
            input,
            num_filters1=459,
            num_filters2=477,
            num_groups=459,
            stride=1,
            scale=scale,
            name="conv5" + "_" + str(5))

        # 7x7
        input = self.depthwise_separable(
            input,
            num_filters1=477,
            num_filters2=967,
            num_groups=477,
            stride=2,
            scale=scale,
            name="conv5_6")

        input = self.depthwise_separable(
            input,
            num_filters1=967,
            num_filters2=516,
            num_groups=967,
            stride=1,
            scale=scale,
            name="conv6")

        input = fluid.layers.pool2d(
            input=input,
            pool_size=0,
            pool_stride=1,
            pool_type='avg',
            global_pooling=True)
        with fluid.name_scope('last_fc'):
            output = fluid.layers.fc(input=input,
                                     size=class_dim,
                                     param_attr=ParamAttr(
                                         initializer=MSRA(),
                                         name="fc7_weights"),
                                     bias_attr=ParamAttr(name="fc7_offset"))

        return output


class MobileNetSensitive50(MobileNet):
    def net(self, input, class_dim=1000, scale=1.0):
        # conv1: 112x112
        input = self.conv_bn_layer(
            input,
            filter_size=3,
            channels=3,
            num_filters=int(32 * scale),
            stride=2,
            padding=1,
            name="conv1")

        # 56x56
        input = self.depthwise_separable(
            input,
            num_filters1=32,
            num_filters2=34,
            num_groups=32,
            stride=1,
            scale=scale,
            name="conv2_1")

        input = self.depthwise_separable(
            input,
            num_filters1=34,
            num_filters2=42,
            num_groups=34,
            stride=2,
            scale=scale,
            name="conv2_2")

        # 28x28
        input = self.depthwise_separable(
            input,
            num_filters1=42,
            num_filters2=116,
            num_groups=42,
            stride=1,
            scale=scale,
            name="conv3_1")

        input = self.depthwise_separable(
            input,
            num_filters1=116,
            num_filters2=138,
            num_groups=116,
            stride=2,
            scale=scale,
            name="conv3_2")

        # 14x14
        input = self.depthwise_separable(
            input,
            num_filters1=138,
            num_filters2=221,
            num_groups=138,
            stride=1,
            scale=scale,
            name="conv4_1")

        input = self.depthwise_separable(
            input,
            num_filters1=221,
            num_filters2=307,
            num_groups=221,
            stride=2,
            scale=scale,
            name="conv4_2")

        # 14x14
        input = self.depthwise_separable(
            input,
            num_filters1=307,
            num_filters2=473,
            num_groups=307,
            stride=1,
            scale=scale,
            name="conv5" + "_" + str(1))

        input = self.depthwise_separable(
            input,
            num_filters1=473,
            num_filters2=475,
            num_groups=473,
            stride=1,
            scale=scale,
            name="conv5" + "_" + str(2))

        input = self.depthwise_separable(
            input,
            num_filters1=475,
            num_filters2=428,
            num_groups=475,
            stride=1,
            scale=scale,
            name="conv5" + "_" + str(3))

        input = self.depthwise_separable(
            input,
            num_filters1=428,
            num_filters2=416,
            num_groups=428,
            stride=1,
            scale=scale,
            name="conv5" + "_" + str(4))

        input = self.depthwise_separable(
            input,
            num_filters1=416,
            num_filters2=446,
            num_groups=416,
            stride=1,
            scale=scale,
            name="conv5" + "_" + str(5))

        # 7x7
        input = self.depthwise_separable(
            input,
            num_filters1=446,
            num_filters2=925,
            num_groups=446,
            stride=2,
            scale=scale,
            name="conv5_6")

        input = self.depthwise_separable(
            input,
            num_filters1=925,
            num_filters2=370,
            num_groups=925,
            stride=1,
            scale=scale,
            name="conv6")

        input = fluid.layers.pool2d(
            input=input,
            pool_size=0,
            pool_stride=1,
            pool_type='avg',
            global_pooling=True)
        with fluid.name_scope('last_fc'):
            output = fluid.layers.fc(input=input,
                                     size=class_dim,
                                     param_attr=ParamAttr(
                                         initializer=MSRA(),
                                         name="fc7_weights"),
                                     bias_attr=ParamAttr(name="fc7_offset"))

        return output
