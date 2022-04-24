#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

import logging
import os
import sys
import numpy as np
import warnings

import paddle
###from ..core import GraphWrapper
from paddleslim.common.recover_program import recover_inference_program
from paddleslim.core import GraphWrapper
from transformer_pattern import *
from dynabert import *

global_idx = 0


def _feed_op_num(program):
    num = 0
    for block in program.blocks:
        ops = list(block.ops)
        for op in ops:
            if op.type == "feed":
                num += 1
    return num


def _fetch_op_input(program):
    fetch_list = []
    for block in program.blocks:
        ops = list(block.ops)
        for op in ops:
            if op.type == "fetch":
                fetch_list.extend(op.input_arg_names)
    return fetch_list


def find_final_nodes(program):
    final_nodes = []
    graph = GraphWrapper(program)
    for op in sorted(graph.ops()):
        if op.type() in ALL_WEIGHT_OP and _is_output_weight_ops(op, graph):
            if _has_bias(op, graph) != None:
                final_nodes.append(n_op.all_outputs())
            else:
                if op.type() == 'batch_norm':
                    out_var = op.outputs('Y')
                else:
                    out_var = op.all_outpus()
                final_nodes.append(out_var)
    ###print(final_nodes)
    return final_nodes


def find_next_ops(block, var_name):
    """
    Find all followed ops for the input variable.
    """
    res_ops = []
    for op in block.ops:
        if var_name in op.input_arg_names:
            res_ops.append(op)
    return res_ops


def insert_eltmul_op(block, op, head_mask, block_num):
    op_idx = block.ops.index(op)
    var_name = op.output_arg_names
    for var_name in op.output_arg_names:
        next_op = find_next_ops(block, var_name)
        score_name = var_name
        if len(next_op) > 0:
            break
    next_op = next_op[0]

    ### start to insert matmul op
    score = block.var(score_name)

    matmul_out_var = block.create_var(
        type=score.type,
        name="{}_eltmul_mask".format(score.name),
        shape=score.shape,
        dtype=score.dtype)

    mask = slice_op(block, block_num, head_mask, op_idx + 1)

    inputs = {"X": score, "Y": mask}
    outputs = {"Out": matmul_out_var}
    block._insert_op(
        op_idx + 2, type='elementwise_mul', inputs=inputs, outputs=outputs)
    next_op_new_input = matmul_out_var.name
    next_op._rename_input(score_name, next_op_new_input)


def fill_constant_op(block,
                     op_idx,
                     shape,
                     value,
                     force_cpu=False,
                     out=None,
                     stop_gradient=True):
    block._insert_op(
        op_idx,
        type='fill_constant',
        outputs={'Out': out},
        attrs={
            'shape': shape,
            'dtype': out.dtype,
            'value': float(value),
            'force_cpu': force_cpu
        })
    out.stop_gradient = stop_gradient
    return out


def unsqueeze_op(block, axis, inputs, op_idx):
    out_name = inputs.name
    out_shape = list(inputs.shape)
    out_shape.insert(axis, 1)
    global global_idx
    out = block.create_var(
        name='{}.unsqueeze_out.tmp_{}'.format(out_name, global_idx),
        shape=out_shape,
        dtype=inputs.dtype)
    global_idx += 1
    block._insert_op(
        op_idx,
        type='unsqueeze',
        inputs={'X': inputs},
        outputs={'Out': out},
        attrs={"axes": [axis]})
    return out


def feed_op(block, op_idx, out):
    feed_var = block.var('feed')

    block._prepend_op(
        op_idx,
        type='feed',
        inputs={'X': [feed_var]},
        outputs={'Out': [out]},
        attrs={'col': op_idx})
    return out


def slice_op(block, axis, inputs, op_idx):
    out_name = inputs.name
    out_shape = list(inputs.shape)
    out_shape.pop(0)
    global global_idx
    out = block.create_var(
        name='{}.slice_out.tmp_{}'.format(out_name, global_idx),
        shape=out_shape,
        dtype=inputs.dtype)
    global_idx += 1
    attrs = {
        "axes": [0],
        "starts": [axis],
        "ends": [axis + 1],
        "decrease_axis": [0]
    }
    block._insert_op(
        op_idx,
        type='slice',
        inputs={'Input': inputs},
        attrs=attrs,
        outputs={'Out': out})
    return out


def softmax_with_cross_entropy_op(block, logits, labels):
    global global_idx
    softmax = block.create_var(
        name='{}.sce.softmax_tmp_{}'.format(logits.name, global_idx),
        shape=logits.shape,
        dtype=logits.dtype)
    loss = block.create_var(
        name='{}.sce.loss_tmp_{}'.format(logits.name, global_idx),
        shape=logits.shape,
        dtype=logits.dtype)
    global_idx += 1
    attrs = {
        'soft_label': False,
        'ignore_index': -100,
        'numeric_stable_mode': True,
        'axis': -1
    }
    inputs = {'Logits': logits, 'Label': labels}
    outputs = {'Softmax': softmax, 'Loss': loss}
    block.append_op(
        type='softmax_with_cross_entropy',
        inputs=inputs,
        outputs=outputs,
        attrs=attrs)
    return loss, softmax


def mean_op(block, inputs, axis=None, keepdim=False):
    global global_idx

    if isinstance(axis, int):
        axis = [axis]
    reduce_all = True if axis is None \
        or len(axis)==0 \
        or len(axis) == len(x.shape) else False
    if axis is None or len(axis) == 0:
        axis = [0]

    if reduce_all == True:
        out_shape = [1]
    else:
        out_shape = list(inputs.shape)
        for idx in sorted(axis, reverse=True):
            out_shape.pop(idx)

    out = block.create_var(
        name='{}.mean_tmp_{}'.format(inputs.name, global_idx),
        shape=out_shape,
        dtype=inputs.dtype)
    attrs = {'dim': axis, 'keep_dim': keepdim, 'reduce_all': reduce_all}
    block.append_op(
        type='reduce_mean',
        inputs={'X': inputs},
        outputs={'Out': out},
        attrs=attrs)
    return out


def program_add_mask(program, patterns, layer_num, head_num):
    fetch_list = _fetch_op_input(program)
    ###print("fetch_list: ", fetch_list)
    program = recover_inference_program(program)
    block = program.global_block()
    head_mask = block.create_var(
        name='head_mask',
        shape=[layer_num, head_num],
        dtype='float32',
        persistable=True)
    feed_num = _feed_op_num(program)
    fill_constant_op(
        block,
        feed_num, [layer_num, head_num],
        1.0,
        out=head_mask,
        stop_gradient=False)
    head_mask = unsqueeze_op(
        block, -1,
        unsqueeze_op(block, -1,
                     unsqueeze_op(block, 1, head_mask, feed_num + 1),
                     feed_num + 2), feed_num + 3)

    #print(head_mask, head_importance)
    for pattern_name, pattern in patterns.items():
        if 'MHA' in pattern_name:
            block_num = int(pattern_name.split('$')[-1])
            for op in pattern:
                if op.type() == 'softmax':
                    var_name = op._op.output_arg_names[0]
                    next_op = find_next_ops(block, var_name)
                    if next_op[0].type == 'dropout':
                        op = next_op[0]
                    insert_eltmul_op(block, op, head_mask, block_num)
    logits = block.var(fetch_list[0])
    labels = block.create_var(
        name='labels', shape=[-1, 1], dtype='int64', persistable=False)
    labels = feed_op(block, feed_num, labels)
    ce_loss, probs = softmax_with_cross_entropy_op(
        block, logits=logits, labels=labels)
    loss = mean_op(block, ce_loss)

    ###print(program)
    program._sync_with_cpp()
    paddle.static.append_backward(loss)
    program._sync_with_cpp()
    ###print(program)
    return program


def compute_importance(program, startup_program, patterns, ffn_weight,
                       layer_num, head_num):
    program = program_add_mask(program, patterns, layer_num, head_num)

    ### define importance matrix
    head_importance = np.zeros(shape=[layer_num, head_num], dtype='float32')
    neuron_importance = []

    intermediate_weight = []
    intermediate_bias = []
    output_weight = []

    fetch_list = ['head_mask@GRAD']
    ### append weight name to fetch list
    for l, wp in ffn_weight.items():
        intermediate_weight.append(wp['P1'][0])
        intermediate_bias.append(wp['P1'][1])
        output_weight.append(wp['P2'][0])
    fetch_list.extend(intermediate_weight)
    fetch_list.extend(intermediate_bias)
    fetch_list.extend(output_weight)

    for out_ws in [intermediate_weight, intermediate_bias, output_weight]:
        for out_w in out_ws:
            fetch_list.append(out_w + '@GRAD')

    ####print("intermediate_weight: ", intermediate_weight)
    ####print("intermediate_bias: ", intermediate_bias)
    ####print("output_weight: ", output_weight)
    print("fetch_list: ", fetch_list)

    for w_name in intermediate_weight:
        neuron_importance.append(
            np.zeros(
                shape=[program.global_block().var(w_name).shape[1]],
                dtype='float32'))

    exe.run(startup_program)
    src_ids = np.random.randint(0, 1, size=[1, 128]).astype('int64')
    sent_ids = np.random.randint(0, 1, size=[1, 128]).astype('int64')
    pos_ids = np.random.randint(0, 1, size=[1, 128]).astype('int64')
    head_mask = np.ones([2, 12]).astype('float32')
    input_mask = np.random.random([1, 128, 1]).astype('float32')
    label = np.ones([1, 1]).astype('int')
    for _ in range(2):
        outs = exe.run(
            program,
            feed={
                #'input_ids': src_ids,
                #'token_type_ids': sent_ids
                'feed_0': src_ids,
                'feed_1': sent_ids,
                'feed_2': pos_ids,
                'feed_3': input_mask,
                'labels': label
            },
            fetch_list=fetch_list)
        #fetch_list=['linear_147.tmp_1'])

        hm_grad_value = outs.pop(0)
        head_importance += np.abs(hm_grad_value)
        part_len = int(len(outs) / 6)
        t_intermediate_weight = outs[:part_len]
        t_intermediate_bias = outs[part_len:2 * part_len]
        t_output_weight = outs[2 * part_len:3 * part_len]
        t_intermediate_weight_grad = outs[3 * part_len:4 * part_len]
        t_intermediate_bias_grad = outs[4 * part_len:5 * part_len]
        t_output_weight_grad = outs[5 * part_len:]
        for w1, w1_g, b1, b1_g, w2, w2_g, current_importance in zip(
                t_intermediate_weight, t_intermediate_weight_grad,
                t_intermediate_bias, t_intermediate_bias_grad, t_output_weight,
                t_output_weight_grad, neuron_importance):
            current_importance += np.abs(
                (np.sum(w1 * w1_g, axis=0) + b1 * b1_g))
            current_importance += np.abs(np.sum(w2 * w2_g, axis=1))

    return program, head_importance, neuron_importance


if __name__ == '__main__':
    paddle.enable_static()
    devices = 'cpu'
    places = paddle.device._convert_to_place(devices)
    exe = paddle.static.Executor(places)
    startup_program = paddle.static.default_startup_program()
    scope = paddle.static.global_scope()
    ###model_dir = '/root/work/source-free/PaddleSlim/demo/auto-compression_origin/MobileNetV2_ssld_infer/'
    ###model_filename = 'inference.pdmodel'
    ###params_filename = 'inference.pdiparams'
    model_dir = '/root/work/test_recompute/'
    model_filename = 'final_export_model.pdmodel'
    params_filename = 'final_export_model.pdiparams'
    ###model_dir = '/root/work/source-free/PaddleSlim/demo/auto-compression_origin/static_bert_models'
    ###model_filename = 'bert.pdmodel'
    ###params_filename = 'bert.pdiparams'
    ###model_dir = '/root/work/source-free/PaddleSlim/paddleslim/auto_compression/picodet_l_640_coco_lcnet_non_postprocess/'
    ######model_dir = '/root/work/source-free/PaddleSlim/paddleslim/auto_compression/ppyoloe_crn_l_300e_coco/'
    ###model_filename = 'model.pdmodel'
    ###params_filename = 'model.pdiparams'
    [inference_program, feed_target_names, fetch_targets]= paddle.fluid.io.load_inference_model( \
        dirname=model_dir, \
        model_filename=model_filename, params_filename=params_filename, \
        executor=exe)

    ### get all patterns
    all_patterns, graph = get_patterns(inference_program, 'transformer')
    ###print(mha_weight, ffn_weight)
    #input_mask_op = all_patterns.pop('input_mask')
    input_mask_op = all_patterns['input_mask']
    layer_num = int((len(all_patterns) - 1) / 2)
    head_num = len(input_mask_op.input_arg_names)

    ###############################     REORDER    ##################################
    compute_program = inference_program.clone()
    ###print(len(all_patterns))
    ###print(all_patterns)
    mha_weight, ffn_weight = preprocess_transformer_patterns(all_patterns,
                                                             graph)
    compute_program, head_importance, neuron_importance = compute_importance(
        compute_program, startup_program, all_patterns, ffn_weight, layer_num,
        head_num)

    reorder_neuron_head(scope, places, mha_weight, ffn_weight, head_importance,
                        neuron_importance, head_num)
    #print(feed_target_names, fetch_targets)
    ###paddle.fluid.io.save_inference_model('test_prog', feed_target_names,
    ###                                     fetch_targets, exe, compute_program)

    ############################### START TO PRUNE ##################################
    mha_weight, ffn_weight = preprocess_transformer_patterns(all_patterns,
                                                             graph)
    ### get input_mask op and start to prune input_mask op
    update_input_mask_inputs(inference_program, input_mask_op, 0.5)

    ### need to send width
    pruned_params = []
    pruned_ratio = []
    for partern in [mha_weight, ffn_weight]:
        for block, part in partern.items():
            pruned_params.extend(part['P1'])
            pruned_ratio.extend(len(part['P1']) * [[1.0, 0.5]])
            pruned_params.extend(part['P2'])
            pruned_ratio.extend(len(part['P2']) * [[0.5, 1.0]])
            if 'reshape_op' in part:
                for op in part['reshape_op']:
                    origin_shape = op.attr('shape')
                    pruned_shape = origin_shape
                    if len(origin_shape) == 3:
                        pruned_shape[-1] = int(origin_shape[-1] * 0.5)
                        op.set_attr('shape', pruned_shape)
                    elif len(origin_shape) == 4:
                        pruned_shape[-2] = int(origin_shape[-2] * 0.5)
                        op.set_attr('shape', pruned_shape)
                    else:
                        raise IndexError
    pruned_dict = dict(zip(pruned_params, pruned_ratio))
    ###print(pruned_dict)
    ### start to prune weight
    program = prune_transformer(scope, places, graph, pruned_dict)
    print(program.global_block().all_parameters())
    ###print(program)

    ### find final node to add distill node
    ###find_final_nodes(inference_program)
