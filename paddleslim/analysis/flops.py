# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License.

import numpy as np
import paddle.fluid.dygraph.jit as jit
from ..core import GraphWrapper

__all__ = ["dygraph_flops", "flops"]


def dygraph_flops(model,
                  inputs,
                  only_conv=True,
                  only_multiply=False,
                  detail=False):
    """
    Get FLOPs of dygraph model.
    Args:
        model: The dygraph model to calculate FLOPs.
        inputs: The inputs of the model, used to calculate FLOPs.
        only_conv(bool): Just return number of mul-adds in convolution and FC layer if `only_conv` is true.
                         Default: True.
        only_multiply(bool): If `only_multiply` is true, just return number of muliply in the model, the 
                         multiply in such as conv, conv_transpose, norm and mul operators will be count.
                         Default: False.
        detail(bool): Whether to return detail of each convolution layer. Default: False.
    """
    _, program, _, _, _ = jit._trace(model, inputs)
    graph = GraphWrapper(program)
    return _graph_flops(
        graph, only_conv=only_conv, only_multiply=only_multiply, detail=detail)


def flops(program, only_conv=True, only_multiply=False, detail=False):
    """
    Get FLOPS of target graph.
    Args:
        program(Program): The program used to calculate FLOPS.
        only_conv(bool): Just return number of mul-adds in convolution and FC layer if `only_conv` is true.
                         default: True.
        only_multiply(bool): Just return number of muliply in the model if `only_multiply` is true.
                         Default: False.
        detail(bool): Whether to return detail of each convolution layer. Default: False.
    
    Return:
        If `detail` is true, then return a tuple in format `(FLOPs, details)`, otherwise it will just return `FlOPs`
        FLOPs(int): The FLOPs of target network.
        details(dict): The key is the parameter name of convlution layer and the value is the FLOPs of each convolution layer.
    """
    graph = GraphWrapper(program)
    return _graph_flops(
        graph, only_conv=only_conv, only_multiply=only_multiply, detail=detail)


def _graph_flops(graph, only_conv=True, only_multiply=False, detail=False):
    assert isinstance(graph, GraphWrapper)
    flops = 0
    params2flops = {}
    for op in graph.ops():
        if op.type() in ['conv2d', 'depthwise_conv2d', 'conv2d_transpose']:
            filter_shape = op.inputs("Filter")[0].shape()
            output_shape = op.outputs("Output")[0].shape()
            if op.type() == 'conv2d_transpose':
                c_in, c_out, k_h, k_w = filter_shape
            else:
                c_out, c_in, k_h, k_w = filter_shape
            _, _, h_out, w_out = output_shape
            # c_in is the channel number of filter. It is (input_channel // groups).
            kernel_ops = k_h * k_w * float(c_in)
            ### after dygraph model to static program, conv op donnot have Bias attrs
            op_flops = h_out * w_out * c_out * kernel_ops
            flops += op_flops
            params2flops[op.inputs("Filter")[0].name()] = op_flops

        elif op.type() == 'pool2d' and not only_conv and not only_multiply:
            output_shape = op.outputs("Out")[0].shape()
            _, c_out, h_out, w_out = output_shape
            k_size = op.attr("ksize")
            op_flops = h_out * w_out * c_out * (k_size[0]**2)
            flops += op_flops

        elif op.type() == 'mul':
            x_shape = list(op.inputs("X")[0].shape())
            y_shape = op.inputs("Y")[0].shape()
            if x_shape[0] == -1:
                x_shape[0] = 1
            flops += x_shape[0] * x_shape[1] * y_shape[1]

            op_flops = x_shape[0] * x_shape[1] * y_shape[1]
            flops += op_flops
            params2flops[op.inputs("Y")[0].name()] = op_flops

        elif op.type() in ['relu', 'sigmoid', 'relu6'
                           ] and not only_conv and not only_multiply:
            input_shape = list(op.inputs("X")[0].shape())
            if input_shape[0] == -1:
                input_shape[0] = 1
            op_flops = np.product(input_shape)
            flops += op_flops

        elif op.type() in ['batch_norm', 'instance_norm', 'layer_norm'
                           ] and not only_conv and only_multiply:
            input_shape = list(op.inputs("X")[0].shape())
            if input_shape[0] == -1:
                input_shape[0] = 1
            op_flops = np.product(input_shape)
            flops += op_flops

        elif op.type() in ['elementwise_add'] and not only_multiply:
            input_shape = list(op.inputs("X")[0].shape())
            ### if inputs Y is parameter that means add bias after conv
            if op.inputs("Y")[0].is_parameter() or not only_conv:
                if input_shape[0] == -1:
                    input_shape[0] = 1
                op_flops = np.product(input_shape)
                flops += op_flops

    if detail:
        return flops, params2flops
    else:
        return flops
