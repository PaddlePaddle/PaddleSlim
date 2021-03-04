#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import numpy as np
import paddle
import paddle.fluid as F
import paddle.fluid.dygraph as FD
import paddle.fluid.layers as L


def compute_neuron_head_importance(args, model, dev_ds, place, model_cfg):
    n_layers, n_heads = model_cfg['num_hidden_layers'], model_cfg[
        'num_attention_heads']
    head_importance = L.zeros(shape=[n_layers, n_heads], dtype='float32')
    head_mask = L.ones(shape=[n_layers, n_heads], dtype='float32')
    head_mask.stop_gradient = False

    intermediate_weight = []
    intermediate_bias = []
    output_weight = []

    for name, w in model.named_parameters():
        if 'ffn.i' in name:
            if len(w.shape) > 1:
                intermediate_weight.append(w)
            else:
                intermediate_bias.append(w)

        if 'ffn.o' in name:
            if len(w.shape) > 1:
                output_weight.append(w)

    neuron_importance = []
    for w in intermediate_weight:
        neuron_importance.append(np.zeros(shape=[w.shape[1]], dtype='float32'))

    eval_task_names = ('mnli', 'mnli-mm') if args.task == 'mnli' else (
        args.task, )

    for eval_task in eval_task_names:
        for batch in dev_ds.start(place):
            ids, sids, label = batch
            out = model(
                ids,
                sids,
                labels=label,
                head_mask=head_mask,
                num_layers=model_cfg['num_hidden_layers'])
            loss = out[0]
            loss.backward()
            head_importance += L.abs(FD.to_variable(head_mask.gradient()))

            for w1, b1, w2, current_importance in zip(
                    intermediate_weight, intermediate_bias, output_weight,
                    neuron_importance):
                current_importance += np.abs(
                    (np.sum(w1.numpy() * w1.gradient(), axis=0) + b1.numpy() *
                     b1.gradient()))
                current_importance += np.abs(
                    np.sum(w2.numpy() * w2.gradient(), axis=1))

    return head_importance, neuron_importance


def reorder_neuron_head(model, head_importance, neuron_importance):
    # reorder heads and ffn neurons
    for layer, current_importance in enumerate(neuron_importance):
        # reorder heads
        idx = L.argsort(head_importance[layer], descending=True)[-1]
        #model.encoder_stack.block[layer].attn.reorder_heads(idx)
        reorder_head(model.encoder_stack.block[layer].attn, idx)
        # reorder neurons
        idx = L.argsort(FD.to_variable(current_importance), descending=True)[-1]
        #model.encoder_stack.block[layer].ffn.reorder_neurons(idx)
        reorder_neuron(model.encoder_stack.block[layer].ffn, idx)


def reorder_head(layer, idx):
    n, a = layer.n_head, layer.d_key
    index = L.reshape(
        L.index_select(
            L.reshape(
                L.arange(
                    0, n * a, dtype='int64'), shape=[n, a]),
            idx,
            dim=0),
        shape=[-1])

    def reorder_head_matrix(linearLayer, index, dim=1):
        W = L.index_select(linearLayer.weight, index, dim=dim).detach()
        if linearLayer.bias is not None:
            if dim == 0:
                b = L.assign(linearLayer.bias).detach()
            else:
                b = L.assign(L.index_select(
                    linearLayer.bias, index, dim=0)).detach()

        linearLayer.weight.stop_gradient = True
        linearLayer.weight.set_value(W)
        linearLayer.weight.stop_gradient = False
        if linearLayer.bias is not None:
            linearLayer.bias.stop_gradient = True
            linearLayer.bias.set_value(b)
            linearLayer.bias.stop_gradient = False

    reorder_head_matrix(
        layer.q.fn if hasattr(layer.q, 'fn') else layer.q, index)
    reorder_head_matrix(
        layer.k.fn if hasattr(layer.k, 'fn') else layer.k, index)
    reorder_head_matrix(
        layer.v.fn if hasattr(layer.v, 'fn') else layer.v, index)
    reorder_head_matrix(
        layer.o.fn if hasattr(layer.o, 'fn') else layer.o, index, dim=0)


def reorder_neuron(layer, index, dim=0):
    def reorder_neurons_matrix(linearLayer, index, dim):
        W = L.index_select(linearLayer.weight, index, dim=dim).detach()
        if linearLayer.bias is not None:
            if dim == 0:
                b = L.assign(linearLayer.bias).detach()
            else:
                b = L.assign(L.index_select(
                    linearLayer.bias, index, dim=0)).detach()
        linearLayer.weight.stop_gradient = True
        linearLayer.weight.set_value(W)
        linearLayer.weight.stop_gradient = False

        if linearLayer.bias is not None:
            linearLayer.bias.stop_gradient = True
            linearLayer.bias.set_value(b)
            linearLayer.bias.stop_gradient = False

    reorder_neurons_matrix(
        layer.i.fn if hasattr(layer.i, 'fn') else layer.i, index, dim=1)
    reorder_neurons_matrix(
        layer.o.fn if hasattr(layer.o, 'fn') else layer.o, index, dim=0)
