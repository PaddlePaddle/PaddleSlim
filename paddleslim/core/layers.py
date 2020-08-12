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

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.dygraph_utils as dygraph_utils
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.fluid.framework import in_dygraph_mode, _varbase_creator
from paddle.fluid.dygraph.nn import InstanceNorm, Conv2D, Conv2DTranspose

from .utils import compute_start_end, get_same_padding, convert_to_list

__all__ = [
    'SuperConv2D', 'SuperConv2DTranspose', 'SuperSeparableConv2D', 'Block'
]

### TODO: add SuperPointConv2D, SuperLinear, SuperBatchNorm
### TODO: complete SuperConv2DTranspose, 


class BaseBlock(fluid.dygraph.Layer):
    def __init__(self, key=None):
        super(BaseBlock, self).__init__()
        if key is not None:
            self._key = str(key)
        else:
            self._key = self.__class__.__key__ + str(unique_name())

    def set_supernet(self, supernet):
        self.__dict__['supernet'] = supernet

    @property
    def key(self):
        return self._key


class Block(BaseBlock):
    #def __init__(self, fn, candidate_config, transform_kernel=None, key=None):
    def __init__(self, fn, transform_kernel=None, key=None):
        super(Block, self).__init__(key)
        #self.transform_kernel = transform_kernel
        self.fn = fn
        self.candidate_config = self.fn.candidate_config

    def forward(self, *inputs, **kwargs):
        out = self.supernet.layers_forward(self, *inputs, **kwargs)
        return out


class SuperConv2D(fluid.dygraph.Conv2D):
    """
    This interface is used to construct a callable object of the ``SuperConv2D``  class.
    The difference between ```SuperConv2D``` and ```Conv2D``` is: ```SuperConv2D``` need 
    to feed a config dictionary with the format of {'channel', num_of_channel} represents 
    the channels of the outputs, used to change the first dimension of weight and bias, 
    only train the first channels of the weight and bias.

    Note: the channel in config need to less than first defined.

    The super convolution2D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input and
    Output are in NCHW format, where N is batch size, C is the number of
    the feature map, H is the height of the feature map, and W is the width of the feature map.
    Filter's shape is [MCHW] , where M is the number of output feature map,
    C is the number of input feature map, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input feature map divided by the groups.
    Please refer to UFLDL's `convolution
    <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_
    for more details.
    If bias attribution and activation type are provided, bias is added to the
    output of the convolution, and the corresponding activation function is
    applied to the final result.
    For each input :math:`X`, the equation is:
    .. math::
        Out = \\sigma (W \\ast X + b)
    Where:
    * :math:`X`: Input value, a ``Tensor`` with NCHW format.
    * :math:`W`: Filter value, a ``Tensor`` with shape [MCHW] .
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D ``Tensor`` with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:
        - Input:
          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`
          Filter shape: :math:`(C_{out}, C_{in}, H_f, W_f)`
        - Output:
          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`
        Where
        .. math::
            H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1
    Parameters:
        num_channels(int): The number of channels in the input image.
        num_filters(int): The number of filter. It is as same as the output
            feature map.
        filter_size (int or tuple): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        stride (int or tuple, optional): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: 1.
        padding (int or tuple, optional): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: 0.
        dilation (int or tuple, optional): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: 1.
        groups (int, optional): The groups number of the Conv2d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: 1.
        param_attr (ParamAttr, optional): The parameter attribute for learnable weights(Parameter)
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with :math:`Normal(0.0, std)`,
            and the :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr or bool, optional): The attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn (bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True.
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            Default: None.
        dtype (str, optional): Data type, it can be "float32" or "float64". Default: "float32".
    Attribute:
        **weight** (Parameter): the learnable weights of filter of this layer.
        **bias** (Parameter or None): the learnable bias of this layer.
    Returns:
        None
    
    Raises:
        ValueError: if ``use_cudnn`` is not a bool value.
    Examples:
        .. code-block:: python
          from paddle.fluid.dygraph.base import to_variable
          import paddle.fluid as fluid
          from paddleslim.core.layers import SuperConv2D
          import numpy as np
          data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
          with fluid.dygraph.guard():
              super_conv2d = SuperConv2D(3, 10, 3)
              config = {'channel': 5}
              data = to_variable(data)
              conv = super_conv2d(data, config)

    """

    ### case1: train max net, config['channels'] = num_filters, config['kernel_size']=filter_size
    ### case2: train super net, config = None, sample sub config from self._progressive_shrinking()
    ### case3: train concert sub net, config['channels']=channels<num_filters, config['filter_size']=kernel_size<filter_size
    ### NOTE: filter_size, num_channels and num_filters must be the max of candidate to define a largest network.
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 candidate_config=None,
                 transform_kernel=None,
                 stride=1,
                 dilation=1,
                 groups=None,
                 param_attr=None,
                 bias_attr=None,
                 use_cudnn=True,
                 act=None,
                 dtype='float32'):
        ### NOTE: padding always is 0, add padding in forward because of kernel size is uncertain
        super(SuperConv2D, self).__init__(
            num_channels, num_filters, filter_size, stride, 0, dilation,
            groups, param_attr, bias_attr, use_cudnn, act, dtype)

        self.candidate_config = candidate_config
        self.filter_size_list = candidate_config[
            'kernel_size'] if 'kernel_size' in candidate_config else self._filter_size
        self.in_channel_list = candidate_config[
            'in_channel_list'] if 'in_channel_list' in candidate_config else self._num_channels
        self.out_channel_list = candidate_config[
            'out_channel_list'] if 'in_channel_list' in candidate_config else self._num_filters
        self.transform_kernel = transform_kernel
        self.ks_set = list(set(self.filter_size_list))
        self.ks_set.sort()
        if self.transform_kernel is not None:
            ### create parameter to transform kernel
            for i in range(len(self.ks_set) - 1):
                ks_small = self.ks_set[i]
                ks_large = self.ks_set[i + 1]
                param_name = '%dto%d_matrix' % (ks_large, ks_small)
                ks_p = ks_small**2
                scale_param[param_name] = self.create_parameter(
                    attr=fluid.ParamAttr(
                        name=param_name,
                        initializer=fluid.initializer.NumpyArrayInitializer(
                            np.eye(ks_p)),
                        shape=(ks_p, ks_p),
                        dtype='float32'))

            for name, param in scale_param.items():
                setattr(self, name, param)

        #### inplace
        #self.weight = self.create_parameter()

    def get_active_filter(self, in_nc, out_nc, kernel_size):
        start, end = compute_start_end(self._filter_size, kernel_size)
        filters = self.weight[:out_nc, :in_nc, start:end, start:end]
        if self.transform_kernel is not None and kernel_size < self._filter_size:
            start_filter = self.weight[:out_nc, :in_nc, :, :]
            for i in range(len(self.ks_set) - 1, 0, -1):
                src_ks = self.ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self.ks_set[i - 1]
                start, end = compute_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = fluid.layers.reshape(
                    _input_filter.shape[0], _input_filter.shape[1], -1)
                _input_filter = fluid.layers.reshape(-1,
                                                     _input_filter.shape[2])
                #matmul_out = _varbase_creator(dtype=_input_filter.dtype)
                core.ops.matmul(_input_filter,
                                self.__getattr__('%dto%d_matrix' %
                                                 (src_ks, target_ks)),
                                _input_filter, 'transpose_X', False,
                                'transpose_Y', False, "alpha", 1)
                _input_filter = fluid.layers.reshape(_input_filter.shape[0],
                                                     _input_filter.shape[1],
                                                     target_ks**2)
                _input_filter = fluid.layers.reshape(_input_filter.shape[0],
                                                     _input_filter.shape[1],
                                                     target_ks, target_ks)
                start_filter = _input_filter
            filters = start_filter
        return filters

    def forward(self, input, kernel_size=None, out_nc=None):  #config=None):
        in_nc = int(input.shape[1])
        out_nc = self._num_filters if out_nc == None else out_nc
        ks = self._filter_size if kernel_size == None else kernel_size
        #out_nc = self.config['channel'] if 'channel' in config else  self._num_filters
        #ks = self.config['kernel_size'] if 'kernel_size' in config else self._filter_size

        ### NOTE: assert in Block
        #assert out_nc <= max(self.out_channel_list), \
        #      "output channel must less than or equal to the max candidate output channel"
        #assert ks <= max(self.filter_size_list), \
        #      "kernel size must less than or equal to the max candidate kernel size"

        weight = self.get_active_filter(in_nc, out_nc, ks)

        padding = convert_to_list(get_same_padding(ks), 2)

        if in_dygraph_mode():
            if self._l_type == 'conv2d':
                attrs = ('strides', self._stride, 'paddings', padding,
                         'dilations', self._dilation, 'groups', self._groups
                         if self._groups else 1, 'use_cudnn', self._use_cudnn)
                out = core.ops.conv2d(input, weight, *attrs)
            elif self._l_type == 'depthwise_conv2d':
                attrs = ('strides', self._stride, 'paddings', padding,
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
            'paddings': padding,
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
    """
    This interface is used to construct a callable object of the ``SuperConv2DTranspose`` 
    class.
    The difference between ```SuperConv2DTranspose``` and ```Conv2DTranspose``` is: 
    ```SuperConv2DTranspose``` need to feed a config dictionary with the format of 
    {'channel', num_of_channel} represents the channels of the outputs, used to change 
    the first dimension of weight and bias, only train the first channels of the weight 
    and bias.

    Note: the channel in config need to less than first defined.

    The super convolution2D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input and output
    are in NCHW format. Where N is batch size, C is the number of feature map,
    H is the height of the feature map, and W is the width of the feature map.
    Filter's shape is [MCHW] , where M is the number of input feature map,
    C is the number of output feature map, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input feature map divided by the groups.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.
    The details of convolution transpose layer, please refer to the following explanation and references
    `conv2dtranspose <http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf>`_ .
    For each input :math:`X`, the equation is:
    .. math::
        Out = \sigma (W \\ast X + b)
    Where:
    * :math:`X`: Input value, a ``Tensor`` with NCHW format.
    * :math:`W`: Filter value, a ``Tensor`` with shape [MCHW] .
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D ``Tensor`` with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.
    Example:
        - Input:
          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`
          Filter shape: :math:`(C_{in}, C_{out}, H_f, W_f)`
        - Output:
          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`
        Where
        .. math::
           H^\prime_{out} &= (H_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (H_f - 1) + 1 \\\\
           W^\prime_{out} &= (W_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (W_f - 1) + 1 \\\\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[0] ) \\\\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[1] )
    Parameters:
        num_channels(int): The number of channels in the input image.
        num_filters(int): The number of the filter. It is as same as the output
            feature map.
        filter_size(int or tuple): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        output_size(int or tuple, optional): The output image size. If output size is a
            tuple, it must contain two integers, (image_H, image_W). None if use
            filter_size, padding, and stride to calculate output_size.
            if output_size and filter_size are specified at the same time, They
            should follow the formula above. Default: None.
        padding(int or tuple, optional): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: 0.
        stride(int or tuple, optional): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: 1.
        dilation(int or tuple, optional): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: 1.
        groups(int, optional): The groups number of the Conv2d transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: 1.
        param_attr (ParamAttr, optional): The parameter attribute for learnable weights(Parameter)
            of conv2d_transpose. If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr or bool, optional): The attribute for the bias of conv2d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn(bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True.
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            Default: None.
        dtype (str, optional): Data type, it can be "float32" or "float64". Default: "float32".
    Attribute:
        **weight** (Parameter): the learnable weights of filters of this layer.
        **bias** (Parameter or None): the learnable bias of this layer.
    Returns:
        None
    Examples:
       .. code-block:: python
          import paddle.fluid as fluid
          from paddleslim.core.layers import SuperConv2DTranspose
          import numpy as np
          with fluid.dygraph.guard():
              data = np.random.random((3, 32, 32, 5)).astype('float32')
              config = {'channel': 5
              super_convtranspose = SuperConv2DTranspose(num_channels=32, num_filters=10, filter_size=3)
              ret = super_convtranspose(fluid.dygraph.base.to_variable(data), config)
    """

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

    def forward(self, input, config=None):
        in_nc = int(input.shape[1])
        out_nc = int(config['channel']) if config else self._num_filters
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
    """
    This interface is used to construct a callable object of the ``SuperSeparableConv2D``
    class.
    The difference between ```SuperSeparableConv2D``` and ```SeparableConv2D``` is: 
    ```SuperSeparableConv2D``` need to feed a config dictionary with the format of 
    {'channel', num_of_channel} represents the channels of the first conv's outputs and
    the second conv's inputs, used to change the first dimension of weight and bias, 
    only train the first channels of the weight and bias.

    The architecture of super separable convolution2D op is [Conv2D, norm layer(may be BatchNorm
    or InstanceNorm), Conv2D]. The first conv is depthwise conv, the filter number is input channel
    multiply scale_factor, the group is equal to the number of input channel. The second conv
    is standard conv, which filter size and stride size are 1. 

    Parameters:
        num_channels(int): The number of channels in the input image.
        num_filters(int): The number of the second conv's filter. It is as same as the output
            feature map.
        filter_size(int or tuple): The first conv's filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        padding(int or tuple, optional): The first conv's padding size. If padding is a tuple, 
            it must contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: 0.
        stride(int or tuple, optional): The first conv's stride size. If stride is a tuple,
            it must contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: 1.
        dilation(int or tuple, optional): The first conv's dilation size. If dilation is a tuple, 
            it must contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: 1.
        norm_layer(class): The normalization layer between two convolution. Default: InstanceNorm.
        bias_attr (ParamAttr or bool, optional): The attribute for the bias of convolution.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, convolution
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        scale_factor(float): The scale factor of the first conv's output channel. Default: 1.
        use_cudnn(bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True.
    Returns:
        None
    """

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

        self.conv.extend([norm_layer(num_channels * scale_factor)])

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
