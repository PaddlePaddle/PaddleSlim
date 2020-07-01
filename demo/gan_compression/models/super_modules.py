import paddle.fluid as fluid
import paddle.fluid.dygraph_utils as dygraph_utils
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.dygraph.nn import InstanceNorm, Conv2D, Conv2DTranspose
import paddle.fluid.core as core
import numpy as np

use_cudnn = False


class SuperInstanceNorm(fluid.dygraph.InstanceNorm):
    def __init__(self,
                 num_channels,
                 epsilon=1e-5,
                 param_attr=None,
                 bias_attr=None,
                 dtype='float32'):
        super(SuperInstanceNorm, self).__init__(
            num_channels,
            epsilon=1e-5,
            param_attr=None,
            bias_attr=None,
            dtype='float32')

    def forward(self, input):
        in_nc = int(input.shape[1])
        scale = self.scale[:in_nc]
        bias = self.scale[:in_nc]
        if in_dygraph_mode():
            out, _, _ = core.ops.instance_norm(input, scale, bias, 'epsilon',
                                               self._epsilon)
            return out
        check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                                 "SuperInstanceNorm")

        attrs = {"epsilon": self._epsilon}

        inputs = {"X": [input], "Scale": [scale], "Bias": [bias]}

        saved_mean = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        saved_variance = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        instance_norm_out = self._helper.create_variable_for_type_inference(
            self._dtype)

        outputs = {
            "Y": [instance_norm_out],
            "SavedMean": [saved_mean],
            "SavedVariance": [saved_variance]
        }

        self._helper.append_op(
            type="instance_norm", inputs=inputs, outputs=outputs, attrs=attrs)
        return instance_norm_out


class SuperConv2D(fluid.dygraph.Conv2D):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=None,
                 param_attr=None,
                 bias_attr=None,
                 use_cudnn=True,
                 act=None,
                 dtype='float32'):
        super(SuperConv2D, self).__init__(
            num_channels, num_filters, filter_size, stride, padding, dilation,
            groups, param_attr, bias_attr, use_cudnn, act, dtype)

    def forward(self, input, config):
        in_nc = int(input.shape[1])
        out_nc = config['channel']
        weight = self.weight[:out_nc, :in_nc, :, :]
        #print('super conv shape', weight.shape)
        if in_dygraph_mode():
            if self._l_type == 'conv2d':
                attrs = ('strides', self._stride, 'paddings', self._padding,
                         'dilations', self._dilation, 'groups', self._groups
                         if self._groups else 1, 'use_cudnn', self._use_cudnn)
                out = core.ops.conv2d(input, weight, *attrs)
            elif self._l_type == 'depthwise_conv2d':
                attrs = ('strides', self._stride, 'paddings', self._padding,
                         'dilations', self._dilation, 'groups', self._groups,
                         'use_cudnn', self._use_cudnn)
                out = core.ops.depthwise_conv2d(input, weight, *attrs)
            else:
                raise ValueError("conv type error")

            pre_bias = out
            if self.bias is not None:
                bias = self.bias[:out_nc]
                pre_act = dygraph_utils._append_bias_in_dygraph(pre_bias, bias,
                                                                1)
            else:
                pre_act = pre_bias

            return dygraph_utils._append_activation_in_dygraph(pre_act,
                                                               self._act)

        inputs = {'Input': [input], 'Filter': [weight]}
        attrs = {
            'strides': self._stride,
            'paddings': self._padding,
            'dilations': self._dilation,
            'groups': self._groups if self._groups else 1,
            'use_cudnn': self._use_cudnn,
            'use_mkldnn': False,
        }
        check_variable_and_dtype(
            input, 'input', ['float16', 'float32', 'float64'], 'SuperConv2D')
        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)

        self._helper.append_op(
            type=self._l_type,
            inputs={
                'Input': input,
                'Filter': weight,
            },
            outputs={"Output": pre_bias},
            attrs=attrs)

        if self.bias is not None:
            bias = self.bias[:out_nc]
            pre_act = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [bias]},
                outputs={'Out': [pre_act]},
                attrs={'axis': 1})
        else:
            pre_act = pre_bias

        # Currently, we don't support inplace in dygraph mode
        return self._helper.append_activation(pre_act, act=self._act)


class SuperConv2DTranspose(fluid.dygraph.Conv2DTranspose):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 output_size=None,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=None,
                 param_attr=None,
                 bias_attr=None,
                 use_cudnn=True,
                 act=None,
                 dtype='float32'):
        super(SuperConv2DTranspose,
              self).__init__(num_channels, num_filters, filter_size,
                             output_size, padding, stride, dilation, groups,
                             param_attr, bias_attr, use_cudnn, act, dtype)

    def forward(self, input, config):
        in_nc = int(input.shape[1])
        out_nc = int(config['channel'])
        weight = self.weight[:in_nc, :out_nc, :, :]
        if in_dygraph_mode():
            op = getattr(core.ops, self._op_type)
            out = op(input, weight, 'output_size', self._output_size,
                     'strides', self._stride, 'paddings', self._padding,
                     'dilations', self._dilation, 'groups', self._groups,
                     'use_cudnn', self._use_cudnn)
            pre_bias = out
            if self.bias is not None:
                bias = self.bias[:out_nc]
                pre_act = dygraph_utils._append_bias_in_dygraph(pre_bias, bias,
                                                                1)
            else:
                pre_act = pre_bias

            return dygraph_utils._append_activation_in_dygraph(
                pre_act, act=self._act)

        check_variable_and_dtype(input, 'input',
                                 ['float16', 'float32', 'float64'],
                                 "SuperConv2DTranspose")

        inputs = {'Input': [input], 'Filter': [weight]}
        attrs = {
            'output_size': self._output_size,
            'strides': self._stride,
            'paddings': self._padding,
            'dilations': self._dilation,
            'groups': self._groups,
            'use_cudnn': self._use_cudnn
        }

        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=input.dtype)
        self._helper.append_op(
            type=self._op_type,
            inputs=inputs,
            outputs={'Output': pre_bias},
            attrs=attrs)

        if self.bias is not None:
            pre_act = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [bias]},
                outputs={'Out': [pre_act]},
                attrs={'axis': 1})
        else:
            pre_act = pre_bias

        out = self._helper.append_activation(pre_act, act=self._act)
        return out


class SuperSeparableConv2D(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 norm_layer=InstanceNorm,
                 bias_attr=None,
                 scale_factor=1,
                 use_cudnn=False):
        super(SuperSeparableConv2D, self).__init__()
        self.conv = fluid.dygraph.LayerList([
            fluid.dygraph.nn.Conv2D(
                num_channels=num_channels,
                num_filters=num_channels * scale_factor,
                filter_size=filter_size,
                stride=stride,
                padding=padding,
                use_cudnn=False,
                groups=num_channels,
                bias_attr=bias_attr)
        ])
        if norm_layer == InstanceNorm:
            #self.conv.extend([SuperInstanceNorm(num_channels * scale_factor)])
            self.conv.extend([
                SuperInstanceNorm(
                    num_channels * scale_factor,
                    param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Constant(1.0),
                        learning_rate=0.0,
                        trainable=False),
                    bias_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Constant(0.0),
                        learning_rate=0.0,
                        trainable=False))
            ])
        else:
            raise NotImplementedError
        self.conv.extend([
            Conv2D(
                num_channels=num_channels * scale_factor,
                num_filters=num_filters,
                filter_size=1,
                stride=1,
                use_cudnn=use_cudnn,
                bias_attr=bias_attr)
        ])

    def forward(self, input, config):
        in_nc = int(input.shape[1])
        out_nc = int(config['channel'])
        weight = self.conv[0].weight[:in_nc]
        ###  conv1
        if in_dygraph_mode():
            if self.conv[0]._l_type == 'conv2d':
                attrs = ('strides', self.conv[0]._stride, 'paddings',
                         self.conv[0]._padding, 'dilations',
                         self.conv[0]._dilation, 'groups', in_nc, 'use_cudnn',
                         self.conv[0]._use_cudnn)
                out = core.ops.conv2d(input, weight, *attrs)
            elif self.conv[0]._l_type == 'depthwise_conv2d':
                attrs = ('strides', self.conv[0]._stride, 'paddings',
                         self.conv[0]._padding, 'dilations',
                         self.conv[0]._dilation, 'groups', in_nc, 'use_cudnn',
                         self.conv[0]._use_cudnn)
                out = core.ops.depthwise_conv2d(input, weight, *attrs)
            else:
                raise ValueError("conv type error")

            pre_bias = out
            if self.conv[0].bias is not None:
                bias = self.conv[0].bias[:in_nc]
                pre_act = dygraph_utils._append_bias_in_dygraph(pre_bias, bias,
                                                                1)
            else:
                pre_act = pre_bias

            conv0_out = dygraph_utils._append_activation_in_dygraph(
                pre_act, self.conv[0]._act)

        norm_out = self.conv[1](conv0_out)

        weight = self.conv[2].weight[:out_nc, :in_nc, :, :]

        if in_dygraph_mode():
            if self.conv[2]._l_type == 'conv2d':
                attrs = ('strides', self.conv[2]._stride, 'paddings',
                         self.conv[2]._padding, 'dilations',
                         self.conv[2]._dilation, 'groups', self.conv[2]._groups
                         if self.conv[2]._groups else 1, 'use_cudnn',
                         self.conv[2]._use_cudnn)
                out = core.ops.conv2d(norm_out, weight, *attrs)
            elif self.conv[2]._l_type == 'depthwise_conv2d':
                attrs = ('strides', self.conv[2]._stride, 'paddings',
                         self.conv[2]._padding, 'dilations',
                         self.conv[2]._dilation, 'groups',
                         self.conv[2]._groups, 'use_cudnn',
                         self.conv[2]._use_cudnn)
                out = core.ops.depthwise_conv2d(norm_out, weight, *attrs)
            else:
                raise ValueError("conv type error")

            pre_bias = out
            if self.conv[2].bias is not None:
                bias = self.conv[2].bias[:out_nc]
                pre_act = dygraph_utils._append_bias_in_dygraph(pre_bias, bias,
                                                                1)
            else:
                pre_act = pre_bias

            conv1_out = dygraph_utils._append_activation_in_dygraph(
                pre_act, self.conv[2]._act)
        return conv1_out


if __name__ == '__main__':

    class Net(fluid.dygraph.Layer):
        def __init__(self, in_cn=3):
            super(Net, self).__init__()
            self.myconv = SuperSeparableConv2D(
                num_channels=in_cn, num_filters=3, filter_size=3)

        def forward(self, input, config):
            print(input.shape[1])
            conv = self.myconv(input, config)

            return conv

    config = {'channel': 2}
    with fluid.dygraph.guard():
        net = Net()

        data_A = np.random.random((1, 3, 256, 256)).astype("float32")
        data_A = to_variable(data_A)

        out = net(data_A, config)
        print(out.numpy())
