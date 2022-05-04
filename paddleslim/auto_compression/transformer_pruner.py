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

import numpy as np
import paddle
from paddleslim.common.recover_program import recover_inference_program
from paddleslim.core import GraphWrapper
from .transformer_pattern import preprocess_transformer_patterns

global_idx = 0


### start to create trainable program with head mask
def _feed_op_num(program):
    num = 0
    for block in program.blocks:
        ops = list(block.ops)
        for op in ops:
            if op.type == "feed":
                num += 1
    return num


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
        or len(axis) == len(inputs.shape) else False
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


class TransformerPruner:
    def __init__(self, exe, places, inference_program, patterns, label_name,
                 width_mult, fetch_targets, dataloader):
        self.exe = exe
        self.places = places
        self.inference_program = inference_program
        self.graph = GraphWrapper(inference_program)
        self.patterns = patterns
        self.label_name = label_name
        self.width_mult = width_mult
        self.fetch_targets = fetch_targets
        self.dataloader = dataloader

        self.scope = paddle.static.global_scope()
        input_mask_op, layer_num, head_num, mha_weight, ffn_weight = self._preprocess_patterns(
            patterns, self.graph)
        self.input_mask_op = input_mask_op
        self.mha_weight = mha_weight
        self.ffn_weight = ffn_weight

        self.scope = self.reorder(inference_program, self.scope, patterns,
                                  layer_num, head_num, mha_weight, ffn_weight)

    def _preprocess_patterns(self, patterns, graph):
        input_mask_op = patterns['input_mask']
        layer_num = int((len(patterns) - 1) / 2)
        head_num = len(input_mask_op.input_arg_names)

        mha_weight, ffn_weight = preprocess_transformer_patterns(patterns,
                                                                 graph)
        return input_mask_op, layer_num, head_num, mha_weight, ffn_weight

    def _program_add_mask(self, program, patterns, layer_num, head_num,
                          label_name, fetch_targets):
        fetch_list = []
        for ft in fetch_targets:
            fetch_list.append(ft.name)
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
            name=label_name, shape=[-1, 1], dtype='int64', persistable=False)
        labels = feed_op(block, feed_num, labels)
        ce_loss, probs = softmax_with_cross_entropy_op(
            block, logits=logits, labels=labels)
        loss = mean_op(block, ce_loss)

        program._sync_with_cpp()
        paddle.static.append_backward(loss)
        program._sync_with_cpp()
        return program

    def compute_importance(self, exe, program, patterns, ffn_weight, layer_num,
                           head_num, label_name, fetch_targets, dataloader):
        program = self._program_add_mask(program, patterns, layer_num, head_num,
                                         label_name, fetch_targets)

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

        for w_name in intermediate_weight:
            neuron_importance.append(
                np.zeros(
                    shape=[program.global_block().var(w_name).shape[1]],
                    dtype='float32'))

        exe.run(paddle.static.default_startup_program())

        ### need to send a dataloader with label
        for batch_id, data in enumerate(dataloader()):
            outs = exe.run(program, feed=data, fetch_list=fetch_list)

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
                    t_intermediate_bias, t_intermediate_bias_grad,
                    t_output_weight, t_output_weight_grad, neuron_importance):
                current_importance += np.abs(
                    (np.sum(w1 * w1_g, axis=0) + b1 * b1_g))
                current_importance += np.abs(np.sum(w2 * w2_g, axis=1))

        return program, head_importance, neuron_importance

    ### REORDER
    def _reorder_head(self, scope, place, weight, head_num, idx):
        qkv = weight['P1']
        attn_out = weight['P2']
        attn_out_t = scope.find_var(qkv[0]).get_tensor()
        num_per_head = int(attn_out_t.shape()[0] / head_num)

        index = np.reshape(
            np.take(
                np.reshape(
                    np.arange(
                        0, head_num * num_per_head, dtype='int64'),
                    (head_num, num_per_head)),
                idx,
                axis=0), (-1))

        def reorder_head_matrix(w_name, index, dim):
            pd_w = scope.find_var(w_name).get_tensor()
            np_w = np.array(pd_w)

            new_w = np.take(np_w, index, axis=dim)
            pd_w.set(new_w, place)

        for w_idx, weight_name in enumerate(qkv):
            if w_idx % 2 == 0:
                ### reorder qkv weight 
                reorder_head_matrix(weight_name, index, dim=1)
            else:
                ### reorder qkv bias 
                reorder_head_matrix(weight_name, index, dim=0)

        ### reorder attention output weight 
        reorder_head_matrix(attn_out[0], index, dim=0)

    def _reorder_neuron(self, scope, place, weight, idx):
        ffn_i = weight['P1']
        ffn_o = weight['P2']

        def reorder_neurons_matrix(w_name, index, dim):
            pd_w = scope.find_var(w_name).get_tensor()
            np_w = np.array(pd_w)

            new_w = np.take(np_w, index, axis=dim)
            pd_w.set(new_w, place)

        reorder_neurons_matrix(ffn_i[0], idx, dim=1)
        reorder_neurons_matrix(ffn_i[1], idx, dim=0)
        reorder_neurons_matrix(ffn_o[0], idx, dim=0)

    def reorder_neuron_head(self, scope, place, mha_weight, ffn_weight,
                            head_importance, neuron_importance, head_num):
        for layer, current_importance in enumerate(neuron_importance):
            ### reorder heads
            idx = np.argsort(head_importance[layer])[::-1]
            self._reorder_head(scope, place, mha_weight[layer], head_num, idx)
            ### reorder neurons
            idx = np.argsort(current_importance)[::-1]
            self._reorder_neuron(scope, place, ffn_weight[layer], idx)

    def reorder(self, inference_program, scope, patterns, layer_num, head_num,
                mha_weight, ffn_weight):
        compute_program = inference_program.clone()

        ###########################  COMPUTE IMPORTANCE  ################################
        compute_program, head_importance, neuron_importance = self.compute_importance(
            self.exe, compute_program, patterns, ffn_weight, layer_num,
            head_num, self.label_name, self.fetch_targets, self.dataloader)

        ###############################     REORDER    ##################################
        self.reorder_neuron_head(scope, self.places, mha_weight, ffn_weight,
                                 head_importance, neuron_importance, head_num)

        return scope

    ### PRUNE
    def _update_input_mask_inputs(self, program, op, new_inputs_len):
        input_var_name = op.input_arg_names
        block = program.blocks[0]
        var = block.var(input_var_name[0])
        op.desc.set_input('X', input_var_name[:int(len(input_var_name) * 0.5)])

    def _prune_weight(self, graph, scope, place, pruned_name, pruned_ratio):
        param = graph.var(pruned_name)
        _var = scope.find_var(param.name())
        if _var is None:
            return
        param_t = _var.get_tensor()
        pruned_ratio = [pruned_ratio[1]] if len(param_t.shape(
        )) == 1 else pruned_ratio
        pruned_shape = np.multiply(param_t.shape(), pruned_ratio)
        pruned_shape = list(map(int, pruned_shape))
        param.set_shape(pruned_shape)
        if len(pruned_shape) == 2:
            pruned_param = np.array(param_t)[:pruned_shape[0], :pruned_shape[1]]
        else:
            pruned_param = np.array(param_t)[:pruned_shape[0]]
        param_t.set(pruned_param, place)

    def _prune_transformer(self, scope, place, graph, pruned_dict):
        for name, value in pruned_dict.items():
            ### prune weight
            self._prune_weight(graph, scope, place, name, value)
        graph.infer_shape()
        return graph.program

    def prune(self):
        ### get input_mask op and start to prune input_mask op
        if self.input_mask_op.type == 'stack':
            self._update_input_mask_inputs(self.inference_program,
                                           self.input_mask_op, self.width_mult)

        pruned_params = []
        pruned_ratio = []
        for partern_weight in [self.mha_weight, self.ffn_weight]:
            for block, part in partern_weight.items():
                pruned_params.extend(part['P1'])
                pruned_ratio.extend(len(part['P1']) * [[1.0, self.width_mult]])
                pruned_params.extend(part['P2'])
                pruned_ratio.extend(len(part['P2']) * [[self.width_mult, 1.0]])
                if 'reshape_op' in part:
                    for op in part['reshape_op']:
                        origin_shape = op.attr('shape')
                        pruned_shape = origin_shape
                        if len(origin_shape) == 3:
                            pruned_shape[-1] = int(origin_shape[-1] *
                                                   self.width_mult)
                            op.set_attr('shape', pruned_shape)
                        elif len(origin_shape) == 4:
                            pruned_shape[-2] = int(origin_shape[-2] *
                                                   self.width_mult)
                            op.set_attr('shape', pruned_shape)
                        else:
                            raise IndexError
        pruned_dict = dict(zip(pruned_params, pruned_ratio))

        ### start to prune weight
        pruned_program = self._prune_transformer(self.scope, self.places,
                                                 self.graph, pruned_dict)
        return pruned_program
