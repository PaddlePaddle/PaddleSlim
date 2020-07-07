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

__all__ = ["dygraph_flops", "flops", "dygrpah_macs", "macs"]


def dygraph_flops(model,
                  inputs,
                  only_conv=True,
                  detail=False):
    """
    Get FLOPs of dygraph model.
    Args:
        model: The dygraph model to calculate FLOPs.
        inputs: The inputs of the model, used to calculate FLOPs.
        only_conv(bool): Just return number of mul-adds in convolution and FC layer if `only_conv` is true.
                         Default: True.
        detail(bool): Whether to return detail of each convolution layer. Default: False.
    """
    _, program, _, _, _ = jit._trace(model, inputs)
    graph = GraphWrapper(program)
    return _graph_cals(
        graph, only_conv=only_conv, count='FLOPs', detail=detail)


def flops(program, only_conv=True, detail=False):
    """
    Get FLOPS of target graph.
    Args:
        program(Program): The program used to calculate FLOPS.
        only_conv(bool): Just return number of mul-adds in convolution and FC layer if `only_conv` is true.
                         default: True.
        detail(bool): Whether to return detail of each convolution layer. Default: False.
    
    Return:
        If `detail` is true, then return a tuple in format `(FLOPs, details)`, otherwise it will just return `FlOPs`
        FLOPs(int): The FLOPs of target network.
        details(dict): The key is the parameter name of convlution layer and the value is the FLOPs of each convolution layer.
    """
    graph = GraphWrapper(program)
    return _graph_cals(
        graph, only_conv=only_conv, count='FLOPs', detail=detail)


def dygraph_macs(model,
                  inputs,
                  only_conv=False,
                  detail=False):
    """
    Get FLOPs of dygraph model.
    Args:
        model: The dygraph model to calculate FLOPs.
        inputs: The inputs of the model, used to calculate FLOPs.
        only_conv(bool): Just return number of mul-adds in convolution and FC layer if `only_conv` is true.
                         Default: True.
        detail(bool): Whether to return detail of each convolution layer. Default: False.
    """
    _, program, _, _, _ = jit._trace(model, inputs)
    graph = GraphWrapper(program)
    return _graph_flops(
        graph, only_conv=only_conv, count='MACs', detail=detail)


def macs(program, only_conv=True, detail=False):
    """
    Get FLOPS of target graph.
    Args:
        program(Program): The program used to calculate FLOPS.
        only_conv(bool): Just return number of mul-adds in convolution and FC layer if `only_conv` is true.
                         default: True.
        detail(bool): Whether to return detail of each convolution layer. Default: False.
    
    Return:
        If `detail` is true, then return a tuple in format `(FLOPs, details)`, otherwise it will just return `FlOPs`
        FLOPs(int): The FLOPs of target network.
        details(dict): The key is the parameter name of convlution layer and the value is the FLOPs of each convolution layer.
    """
    graph = GraphWrapper(program)
    return _graph_cals(
        graph, only_conv=only_conv, count='MACs', detail=detail)

def _graph_cals(graph, only_conv=True, count, detail=False):
    assert isinstance(graph, GraphWrapper)
    assert count.lower() in ['flops', 'macs'], "count {} not support now".format(count)
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
            if count == 'MACs':
                out_pixel = k_h * k_w * float(c_in)
            else:
                ### count == 'FLOPs'
                ### add bias count in elementwise_add op
                out_pixel = 2 * k_h * k_w * float(c_in) - 1
                 
            op_flops = h_out * w_out * c_out * out_pixel
            flops += op_flops
            params2flops[op.inputs("Filter")[0].name()] = op_flops

        elif op.type() == 'pool2d' and not only_conv:
            if count == 'MACs':
                op_shape = op.outputs("Out")[0].shape()
            else:
                op_shape = op.inputs("X")[0].shape()
            _, c_out, h_out, w_out = op_shape
            k_size = op.attr("ksize")
            op_flops = h_out * w_out * c_out * (k_size[0]**2)
            flops += op_flops

        elif op.type() == 'mul' and not only_conv:
            x_shape = list(op.inputs("X")[0].shape())
            y_shape = op.inputs("Y")[0].shape()
            if x_shape[0] == -1:
                x_shape[0] = 1

            if count == 'MACs':
                op_flops = x_shape[0] * x_shape[1] * y_shape[1]
            else:
                op_flops = (2 * x_shape[0] * x_shape[1] - 1) * y_shape[1]

            flops += op_flops
            params2flops[op.inputs("Y")[0].name()] = op_flops

        elif op.type() in ['relu', 'sigmoid', 'relu6'
                           ] and not only_conv:
            input_shape = list(op.inputs("X")[0].shape())
            if input_shape[0] == -1:
                input_shape[0] = 1
            op_flops = np.product(input_shape)
            flops += op_flops

        elif op.type() in ['batch_norm', 'instance_norm', 'layer_norm'
                           ] and not only_conv:
            input_shape = list(op.inputs("X")[0].shape())
            if input_shape[0] == -1:
                input_shape[0] = 1
            ### (x - mean) * sqrt(var)
            op_flops = np.product(input_shape)
            if count == 'FLOPs':
                ### NOTE: if scale and bias can be none (Need to fix in bn op), it need to add more condition to determine if need to multiply 2
                op_flops *= 2
            flops += op_flops

        elif op.type() in ['elementwise_add'] and count == 'FLOPs':
            input_shape = list(op.inputs("X")[0].shape())
            ### if inputs Y is parameter that means add bias after conv or norm
            if op.inputs("Y")[0].is_parameter():
                if input_shape[0] == -1:
                    input_shape[0] = 1
                op_flops = np.product(input_shape)
                flops += op_flops

    if detail:
        return flops, params2flops
    else:
        return flops
