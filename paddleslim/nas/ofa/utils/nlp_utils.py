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

import os
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def compute_neuron_head_importance(task_name,
                                   model,
                                   data_loader,
                                   n_layers,
                                   n_heads,
                                   loss_fct=nn.loss.CrossEntropyLoss(),
                                   intermediate_name='linear1',
                                   output_name='linear2'):
    head_importance = paddle.zeros(shape=[n_layers, n_heads], dtype='float32')
    head_mask = paddle.ones(shape=[n_layers, n_heads], dtype='float32')
    head_mask.stop_gradient = False

    intermediate_weight = []
    intermediate_bias = []
    output_weight = []

    for name, w in model.named_parameters():
        if intermediate_name in name:
            if len(w.shape) > 1:
                intermediate_weight.append(w)
            else:
                intermediate_bias.append(w)

        if output_name in name:
            if len(w.shape) > 1:
                output_weight.append(w)

    neuron_importance = []
    for w in intermediate_weight:
        neuron_importance.append(np.zeros(shape=[w.shape[1]], dtype='float32'))

    eval_task_names = ('mnli', 'mnli-mm') if task_name.lower() == 'mnli' else (
        task_name, )

    for eval_task in eval_task_names:
        for batch in data_loader:
            input_ids, segment_ids, labels = batch
            logits = model(
                input_ids, segment_ids, attention_mask=[None, head_mask])
            loss = loss_fct(logits, labels)
            loss.backward()
            head_importance += paddle.abs(
                paddle.to_tensor(head_mask.gradient()))

            for w1, b1, w2, current_importance in zip(
                    intermediate_weight, intermediate_bias, output_weight,
                    neuron_importance):
                current_importance += np.abs(
                    (np.sum(w1.numpy() * w1.gradient(), axis=0) + b1.numpy() *
                     b1.gradient()))
                current_importance += np.abs(
                    np.sum(w2.numpy() * w2.gradient(), axis=1))

    return head_importance, neuron_importance


def reorder_head(layer, idx):
    n, a = layer.num_heads, layer.head_dim
    index = paddle.reshape(
        paddle.index_select(
            paddle.reshape(
                paddle.arange(
                    0, n * a, dtype='int64'), shape=[n, a]),
            index=idx,
            axis=0),
        shape=[-1])

    def reorder_head_matrix(linearLayer, index, dim=1):
        W = paddle.index_select(linearLayer.weight, index, axis=dim).detach()
        if linearLayer.bias is not None:
            if dim == 0:
                b = paddle.assign(linearLayer.bias).detach()
            else:
                b = paddle.assign(
                    paddle.index_select(
                        linearLayer.bias, index, axis=0)).detach()

        linearLayer.weight.stop_gradient = True
        linearLayer.weight.set_value(W)
        linearLayer.weight.stop_gradient = False
        if linearLayer.bias is not None:
            linearLayer.bias.stop_gradient = True
            linearLayer.bias.set_value(b)
            linearLayer.bias.stop_gradient = False

    reorder_head_matrix(layer.q_proj, index)
    reorder_head_matrix(layer.k_proj, index)
    reorder_head_matrix(layer.v_proj, index)
    reorder_head_matrix(layer.out_proj, index, dim=0)


def reorder_neuron(linearLayer, index, dim=0):
    W = paddle.index_select(linearLayer.weight, index, axis=dim).detach()
    if linearLayer.bias is not None:
        if dim == 0:
            b = paddle.assign(linearLayer.bias).detach()
        else:
            b = paddle.assign(
                paddle.index_select(
                    linearLayer.bias, index, axis=0)).detach()
    linearLayer.weight.stop_gradient = True
    linearLayer.weight.set_value(W)
    linearLayer.weight.stop_gradient = False

    if linearLayer.bias is not None:
        linearLayer.bias.stop_gradient = True
        linearLayer.bias.set_value(b)
        linearLayer.bias.stop_gradient = False


### rewrite MultiHeadAttention forward to accept head_mask
def _mha_forward(self, query, key, value, attn_mask=None, cache=None):
    key = query if key is None else key
    value = query if value is None else value
    # compute q ,k ,v
    if cache is None:
        q, k, v = self._prepare_qkv(query, key, value, cache)
    else:
        q, k, v, cache = self._prepare_qkv(query, key, value, cache)

    # scale dot product attention
    # TODO: use paddle.matmul, however it doesn't support `alpha`
    product = paddle.fluid.layers.matmul(
        x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)
    if attn_mask[0] is not None:
        # TODO(guosheng): support bool mask
        product = product + attn_mask[0]
    weights = F.softmax(product)
    if self.dropout:
        weights = F.dropout(
            weights,
            self.dropout,
            training=self.training,
            mode="upscale_in_train")

    if attn_mask[1] is not None:
        weights = weights * attn_mask[1]

    out = paddle.matmul(weights, v)

    # combine heads
    out = paddle.transpose(out, perm=[0, 2, 1, 3])
    out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

    # project to output
    out = self.out_proj(out)

    outs = [out]
    if self.need_weights:
        outs.append(weights)
    if cache is not None:
        outs.append(cache)
    return out if len(outs) == 1 else tuple(outs)


### rewrite TransformerEncoder forward to accept head_mask
def _encoder_forward(self, src, src_mask=[None, None]):
    output = src
    if src_mask[1] is not None:
        head_mask = src_mask[1]
        if len(head_mask.shape) == 1:
            head_mask = paddle.unsqueeze(
                paddle.unsqueeze(
                    paddle.unsqueeze(paddle.unsqueeze(head_mask, 0), 0), -1),
                -1)
            head_mask = paddle.expand(
                head_mask,
                expand_times=[self.cfg['num_hidden_layers'], 1, 1, 1, 1])
        elif len(head_mask.shape) == 2:
            head_mask = paddle.unsqueeze(
                paddle.unsqueeze(paddle.unsqueeze(head_mask, 1), -1), -1)
    else:
        head_mask = [None] * self.num_layers

    for i, mod in enumerate(self.layers):
        output = mod(output, src_mask=[src_mask[0], head_mask[i]])

    if self.norm is not None:
        output = self.norm(output)

    return output


nn.MultiHeadAttention.forward = _mha_forward
nn.TransformerEncoder.forward = _encoder_forward
