import functools
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import BatchNorm, InstanceNorm
from discrimitor import NLayerDiscriminator
from generator.resnet_generator import ResnetGenerator
from generator.mobile_generator import MobileResnetGenerator
from generator.super_generator import SuperMobileResnetGenerator


class Identity(fluid.dygraph.Layer):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'instance':
        norm_layer = functools.partial(
            InstanceNorm,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(1.0),
                learning_rate=0.0,
                trainable=False),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.0),
                learning_rate=0.0,
                trainable=False))
    elif norm_type == 'batch':
        norm_layer = functools.partial(
            BatchNorm,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(1.0, 0.02)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.0)))
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
    return net


def define_D(input_nc, ndf, netD, norm_type='batch', n_layers_D=3):
    net = None
    norm_layer = get_norm_layer(norm_type)
    if netD == 'n_layers':
        net = NLayerDiscriminator(
            input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    return net
