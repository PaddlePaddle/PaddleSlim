from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
import os, sys, time, math
import numpy as np
from collections import namedtuple

BLOCK_TYPE_MCRELU = 'BLOCK_TYPE_MCRELU'
BLOCK_TYPE_INCEP = 'BLOCK_TYPE_INCEP'
BlockConfig = namedtuple('BlockConfig',
                         'stride, num_outputs, preact_bn, block_type')

__all__ = ['PVANet']


class PVANet():
    def __init__(self):
        pass

    def net(self, input, include_last_bn_relu=True, class_dim=1000):
        conv1 = self._conv_bn_crelu(input, 16, 7, stride=2, name="conv1_1")
        pool1 = fluid.layers.pool2d(
            input=conv1,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max',
            name='pool1')

        end_points = {}
        conv2 = self._conv_stage(
            pool1,
            block_configs=[
                BlockConfig(1, (24, 24, 48), False, BLOCK_TYPE_MCRELU),
                BlockConfig(1, (24, 24, 48), True, BLOCK_TYPE_MCRELU),
                BlockConfig(1, (24, 24, 48), True, BLOCK_TYPE_MCRELU)
            ],
            name='conv2',
            end_points=end_points)

        conv3 = self._conv_stage(
            conv2,
            block_configs=[
                BlockConfig(2, (48, 48, 96), True, BLOCK_TYPE_MCRELU),
                BlockConfig(1, (48, 48, 96), True, BLOCK_TYPE_MCRELU),
                BlockConfig(1, (48, 48, 96), True, BLOCK_TYPE_MCRELU),
                BlockConfig(1, (48, 48, 96), True, BLOCK_TYPE_MCRELU)
            ],
            name='conv3',
            end_points=end_points)

        conv4 = self._conv_stage(
            conv3,
            block_configs=[
                BlockConfig(2, '64 48-96 24-48-48 96 128', True,
                            BLOCK_TYPE_INCEP),
                BlockConfig(1, '64 64-96 24-48-48 128', True, BLOCK_TYPE_INCEP),
                BlockConfig(1, '64 64-96 24-48-48 128', True, BLOCK_TYPE_INCEP),
                BlockConfig(1, '64 64-96 24-48-48 128', True, BLOCK_TYPE_INCEP)
            ],
            name='conv4',
            end_points=end_points)

        conv5 = self._conv_stage(
            conv4,
            block_configs=[
                BlockConfig(2, '64 96-128 32-64-64 128 196', True,
                            BLOCK_TYPE_INCEP),
                BlockConfig(1, '64 96-128 32-64-64 196', True,
                            BLOCK_TYPE_INCEP),
                BlockConfig(1, '64 96-128 32-64-64 196', True,
                            BLOCK_TYPE_INCEP),
                BlockConfig(1, '64 96-128 32-64-64 196', True, BLOCK_TYPE_INCEP)
            ],
            name='conv5',
            end_points=end_points)

        if include_last_bn_relu:
            conv5 = self._bn(conv5, 'relu', 'conv5_4_last_bn')
        end_points['conv5'] = conv5

        output = fluid.layers.fc(input=input,
                                 size=class_dim,
                                 param_attr=ParamAttr(
                                     initializer=MSRA(), name="fc_weights"),
                                 bias_attr=ParamAttr(name="fc_offset"))

        return output

    def _conv_stage(self, input, block_configs, name, end_points):
        net = input
        for idx, bc in enumerate(block_configs):
            if bc.block_type == BLOCK_TYPE_MCRELU:
                block_scope = '{}_{}'.format(name, idx + 1)
                fn = self._mCReLU
            elif bc.block_type == BLOCK_TYPE_INCEP:
                block_scope = '{}_{}_incep'.format(name, idx + 1)
                fn = self._inception_block
            net = fn(net, bc, block_scope)
            end_points[block_scope] = net
        end_points[name] = net
        return net

    def _mCReLU(self, input, mc_config, name):
        """
        every cReLU has at least three conv steps:
            conv_bn_relu, conv_bn_crelu, conv_bn_relu
        if the inputs has a different number of channels as crelu output,
        an extra 1x1 conv is added before sum.
        """
        if mc_config.preact_bn:
            conv1_fn = self._bn_relu_conv
            conv1_scope = name + '_1'
        else:
            conv1_fn = self._conv
            conv1_scope = name + '_1_conv'

        sub_conv1 = conv1_fn(input, mc_config.num_outputs[0], 1, conv1_scope,
                             mc_config.stride)

        sub_conv2 = self._bn_relu_conv(sub_conv1, mc_config.num_outputs[1], 3,
                                       name + '_2')

        sub_conv3 = self._bn_crelu_conv(sub_conv2, mc_config.num_outputs[2], 1,
                                        name + '_3')

        if int(input.shape[1]) == mc_config.num_outputs[2]:
            conv_proj = input
        else:
            conv_proj = self._conv(input, mc_config.num_outputs[2], 1,
                                   name + '_proj', mc_config.stride)

        conv = sub_conv3 + conv_proj
        return conv

    def _inception_block(self, input, block_config, name):
        num_outputs = block_config.num_outputs.split()  # e.g. 64 24-48-48 128
        num_outputs = [map(int, s.split('-')) for s in num_outputs]
        inception_outputs = num_outputs[-1][0]
        num_outputs = num_outputs[:-1]
        stride = block_config.stride
        pool_path_outputs = None
        if stride > 1:
            pool_path_outputs = num_outputs[-1][0]
            num_outputs = num_outputs[:-1]

        scopes = [['_0']]  # follow the name style of caffe pva
        kernel_sizes = [[1]]
        for path_idx, path_outputs in enumerate(num_outputs[1:]):
            path_idx += 1
            path_scopes = ['_{}_reduce'.format(path_idx)]
            path_scopes.extend([
                '_{}_{}'.format(path_idx, i - 1)
                for i in range(1, len(path_outputs))
            ])
            scopes.append(path_scopes)

            path_kernel_sizes = [1, 3, 3][:len(path_outputs)]
            kernel_sizes.append(path_kernel_sizes)

        paths = []
        if block_config.preact_bn:
            preact = self._bn(input, 'relu', name + '_bn')
        else:
            preact = input

        path_params = zip(num_outputs, scopes, kernel_sizes)
        for path_idx, path_param in enumerate(path_params):
            path_net = preact
            for conv_idx, (num_output, scope,
                           kernel_size) in enumerate(zip(*path_param)):
                if conv_idx == 0:
                    conv_stride = stride
                else:
                    conv_stride = 1
                path_net = self._conv_bn_relu(path_net, num_output, kernel_size,
                                              name + scope, conv_stride)
            paths.append(path_net)

        if stride > 1:
            path_net = fluid.layers.pool2d(
                input,
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max',
                name=name + '_pool')
            path_net = self._conv_bn_relu(path_net, pool_path_outputs, 1,
                                          name + '_poolproj')
            paths.append(path_net)
        block_net = fluid.layers.concat(paths, axis=1)
        block_net = self._conv(block_net, inception_outputs, 1,
                               name + '_out_conv')

        if int(input.shape[1]) == inception_outputs:
            proj = input
        else:
            proj = self._conv(input, inception_outputs, 1, name + '_proj',
                              stride)
        return block_net + proj

    def _scale(self, input, name, axis=1, num_axes=1):
        assert num_axes == 1, "layer scale not support this num_axes[%d] now" % (
            num_axes)

        prefix = name + '_'
        scale_shape = input.shape[axis:axis + num_axes]
        param_attr = fluid.ParamAttr(name=prefix + 'gamma')
        scale_param = fluid.layers.create_parameter(
            shape=scale_shape,
            dtype=input.dtype,
            name=name,
            attr=param_attr,
            is_bias=True,
            default_initializer=fluid.initializer.Constant(value=1.0))

        offset_attr = fluid.ParamAttr(name=prefix + 'beta')
        offset_param = fluid.layers.create_parameter(
            shape=scale_shape,
            dtype=input.dtype,
            name=name,
            attr=offset_attr,
            is_bias=True,
            default_initializer=fluid.initializer.Constant(value=0.0))

        output = fluid.layers.elementwise_mul(
            input, scale_param, axis=axis, name=prefix + 'mul')
        output = fluid.layers.elementwise_add(
            output, offset_param, axis=axis, name=prefix + 'add')
        return output

    def _conv(self,
              input,
              num_filters,
              filter_size,
              name,
              stride=1,
              groups=1,
              act=None):
        net = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=act,
            use_cudnn=True,
            param_attr=ParamAttr(name=name + '_weights'),
            bias_attr=ParamAttr(name=name + '_bias'),
            name=name)
        return net

    def _bn(self, input, act, name):
        net = fluid.layers.batch_norm(
            input=input,
            act=act,
            name=name,
            moving_mean_name=name + '_mean',
            moving_variance_name=name + '_variance',
            param_attr=ParamAttr(name=name + '_scale'),
            bias_attr=ParamAttr(name=name + '_offset'))
        return net

    def _bn_relu_conv(self,
                      input,
                      num_filters,
                      filter_size,
                      name,
                      stride=1,
                      groups=1):

        net = self._bn(input, 'relu', name + '_bn')
        net = self._conv(net, num_filters, filter_size, name + '_conv', stride,
                         groups)
        return net

    def _conv_bn_relu(self,
                      input,
                      num_filters,
                      filter_size,
                      name,
                      stride=1,
                      groups=1):
        net = self._conv(input, num_filters, filter_size, name + '_conv',
                         stride, groups)
        net = self._bn(net, 'relu', name + '_bn')
        return net

    def _bn_crelu(self, input, name):
        net = self._bn(input, None, name + '_bn_1')
        neg_net = fluid.layers.scale(net, scale=-1.0, name=name + '_neg')
        net = fluid.layers.concat([net, neg_net], axis=1)
        net = self._scale(net, name + '_scale')
        net = fluid.layers.relu(net, name=name + '_relu')
        return net

    def _conv_bn_crelu(self,
                       input,
                       num_filters,
                       filter_size,
                       name,
                       stride=1,
                       groups=1,
                       act=None):
        net = self._conv(input, num_filters, filter_size, name + '_conv',
                         stride, groups)
        net = self._bn_crelu(net, name)
        return net

    def _bn_crelu_conv(self,
                       input,
                       num_filters,
                       filter_size,
                       name,
                       stride=1,
                       groups=1,
                       act=None):
        net = self._bn_crelu(input, name)
        net = self._conv(net, num_filters, filter_size, name + '_conv', stride,
                         groups)
        return net

    def deconv_bn_layer(self,
                        input,
                        num_filters,
                        filter_size=4,
                        stride=2,
                        padding=1,
                        act='relu',
                        name=None):
        """Deconv bn layer."""
        deconv = fluid.layers.conv2d_transpose(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(name=name + '_weights'),
            bias_attr=ParamAttr(name=name + '_bias'),
            name=name + 'deconv')
        return self._bn(deconv, act, name + '_bn')

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      name,
                      stride=1,
                      groups=1):
        return self._conv_bn_relu(input, num_filters, filter_size, name, stride,
                                  groups)


def Fpn_Fusion(blocks, net):
    f = [blocks['conv5'], blocks['conv4'], blocks['conv3'], blocks['conv2']]
    num_outputs = [64] * len(f)
    g = [None] * len(f)
    h = [None] * len(f)
    for i in range(len(f)):
        h[i] = net.conv_bn_layer(f[i], num_outputs[i], 1, 'fpn_pre_' + str(i))

    for i in range(len(f) - 1):
        if i == 0:
            g[i] = net.deconv_bn_layer(h[i], num_outputs[i], name='fpn_0')
        else:
            out = fluid.layers.elementwise_add(x=g[i - 1], y=h[i])
            out = net.conv_bn_layer(out, num_outputs[i], 1,
                                    'fpn_trans_' + str(i))
            g[i] = net.deconv_bn_layer(
                out, num_outputs[i], name='fpn_' + str(i))

    out = fluid.layers.elementwise_add(x=g[-2], y=h[-1])
    out = net.conv_bn_layer(out, num_outputs[-1], 1, 'fpn_post_0')
    out = net.conv_bn_layer(out, num_outputs[-1], 3, 'fpn_post_1')

    return out


def Detector_Header(f_common, net, class_num):
    """Detector header."""
    f_geo = net.conv_bn_layer(f_common, 64, 1, name='geo_1')
    f_geo = net.conv_bn_layer(f_geo, 64, 3, name='geo_2')
    f_geo = net.conv_bn_layer(f_geo, 64, 1, name='geo_3')
    f_geo = fluid.layers.conv2d(
        f_geo,
        8,
        1,
        use_cudnn=True,
        param_attr=ParamAttr(name='geo_4_conv_weights'),
        bias_attr=ParamAttr(name='geo_4_conv_bias'),
        name='geo_4_conv')

    name = 'score_class_num' + str(class_num + 1)
    f_score = net.conv_bn_layer(f_common, 64, 1, 'score_1')
    f_score = net.conv_bn_layer(f_score, 64, 3, 'score_2')
    f_score = net.conv_bn_layer(f_score, 64, 1, 'score_3')
    f_score = fluid.layers.conv2d(
        f_score,
        class_num + 1,
        1,
        use_cudnn=True,
        param_attr=ParamAttr(name=name + '_conv_weights'),
        bias_attr=ParamAttr(name=name + '_conv_bias'),
        name=name + '_conv')

    f_score = fluid.layers.transpose(f_score, perm=[0, 2, 3, 1])
    f_score = fluid.layers.reshape(f_score, shape=[-1, class_num + 1])
    f_score = fluid.layers.softmax(input=f_score)

    return f_score, f_geo


def east(input, class_num=31):
    net = PVANet()
    out = net.net(input)
    blocks = []
    for i, j, k in zip(['conv2', 'conv3', 'conv4', 'conv5'], [1, 2, 4, 8],
                       [64, 64, 64, 64]):
        if j == 1:
            conv = net.conv_bn_layer(
                out[i], k, 1, name='fusion_' + str(len(blocks)))
        elif j <= 4:
            conv = net.deconv_bn_layer(
                out[i], k, 2 * j, j, j // 2, name='fusion_' + str(len(blocks)))
        else:
            conv = net.deconv_bn_layer(
                out[i], 32, 8, 4, 2, name='fusion_' + str(len(blocks)) + '_1')
            conv = net.deconv_bn_layer(
                conv,
                k,
                j // 2,
                j // 4,
                j // 8,
                name='fusion_' + str(len(blocks)) + '_2')
        blocks.append(conv)
    conv = fluid.layers.concat(blocks, axis=1)
    f_score, f_geo = Detector_Header(conv, net, class_num)
    return f_score, f_geo


def inference(input, class_num=1, nms_thresh=0.2, score_thresh=0.5):
    f_score, f_geo = east(input, class_num)
    print("f_geo shape={}".format(f_geo.shape))
    print("f_score shape={}".format(f_score.shape))
    f_score = fluid.layers.transpose(f_score, perm=[1, 0])
    return f_score, f_geo


def loss(f_score, f_geo, l_score, l_geo, l_mask, class_num=1):
    '''
    predictions: f_score: -1 x 1 x H x W; f_geo: -1 x 8 x H x W
    targets: l_score: -1 x 1 x H x W; l_geo: -1 x 1 x H x W; l_mask: -1 x 1 x H x W
    return: dice_loss + smooth_l1_loss
    '''
    #smooth_l1_loss
    channels = 8
    l_geo_split, l_short_edge = fluid.layers.split(
        l_geo, num_or_sections=[channels, 1],
        dim=1)  #last channel is short_edge_norm
    f_geo_split = fluid.layers.split(f_geo, num_or_sections=[channels], dim=1)
    f_geo_split = f_geo_split[0]

    geo_diff = l_geo_split - f_geo_split
    abs_geo_diff = fluid.layers.abs(geo_diff)
    l_flag = l_score >= 1
    l_flag = fluid.layers.cast(x=l_flag, dtype="float32")
    l_flag = fluid.layers.expand(x=l_flag, expand_times=[1, channels, 1, 1])

    smooth_l1_sign = abs_geo_diff < l_flag
    smooth_l1_sign = fluid.layers.cast(x=smooth_l1_sign, dtype="float32")

    in_loss = abs_geo_diff * abs_geo_diff * smooth_l1_sign + (
        abs_geo_diff - 0.5) * (1.0 - smooth_l1_sign)
    l_short_edge = fluid.layers.expand(
        x=l_short_edge, expand_times=[1, channels, 1, 1])
    out_loss = l_short_edge * in_loss * l_flag
    out_loss = out_loss * l_flag
    smooth_l1_loss = fluid.layers.reduce_mean(out_loss)

    ##softmax_loss
    l_score.stop_gradient = True
    l_score = fluid.layers.transpose(l_score, perm=[0, 2, 3, 1])
    l_score.stop_gradient = True
    l_score = fluid.layers.reshape(l_score, shape=[-1, 1])
    l_score.stop_gradient = True
    l_score = fluid.layers.cast(x=l_score, dtype="int64")
    l_score.stop_gradient = True

    softmax_loss = fluid.layers.cross_entropy(input=f_score, label=l_score)
    softmax_loss = fluid.layers.reduce_mean(softmax_loss)

    return softmax_loss, smooth_l1_loss
