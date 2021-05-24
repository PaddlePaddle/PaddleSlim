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

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import math
import argparse
import json
import logging
import logging
from functools import partial
import six
if six.PY2:
    from pathlib2 import Path
else:
    from pathlib import Path

import paddle.fluid.dygraph as D
import paddle.fluid as F
import paddle.fluid.layers as L
from ernie.file_utils import _fetch_from_remote
from ernie.modeling_ernie import AttentionLayer, ErnieBlock, ErnieModel, ErnieEncoderStack, ErnieModelForSequenceClassification

log = logging.getLogger(__name__)


def append_name(name, postfix):
    if name is None:
        return None
    elif name == '':
        return postfix
    else:
        return '%s_%s' % (name, postfix)


def _attn_forward(self,
                  queries,
                  keys,
                  values,
                  attn_bias,
                  past_cache,
                  head_mask=None):
    assert len(queries.shape) == len(keys.shape) == len(values.shape) == 3

    q = self.q(queries)
    k = self.k(keys)
    v = self.v(values)

    cache = (k, v)
    if past_cache is not None:
        cached_k, cached_v = past_cache
        k = L.concat([cached_k, k], 1)
        v = L.concat([cached_v, v], 1)

    if hasattr(self.q, 'fn') and self.q.fn.cur_config['expand_ratio'] != None:
        n_head = int(self.n_head * self.q.fn.cur_config['expand_ratio'])
    else:
        n_head = self.n_head

    q = L.transpose(
        L.reshape(q, [0, 0, n_head, q.shape[-1] // n_head]),
        [0, 2, 1, 3])  #[batch, head, seq, dim]
    k = L.transpose(
        L.reshape(k, [0, 0, n_head, k.shape[-1] // n_head]),
        [0, 2, 1, 3])  #[batch, head, seq, dim]
    v = L.transpose(
        L.reshape(v, [0, 0, n_head, v.shape[-1] // n_head]),
        [0, 2, 1, 3])  #[batch, head, seq, dim]

    q = L.scale(q, scale=self.d_key**-0.5)
    score = L.matmul(q, k, transpose_y=True)
    if attn_bias is not None:
        score += attn_bias

    score = L.softmax(score, use_cudnn=True)
    score = self.dropout(score)
    if head_mask is not None:
        score = score * head_mask

    out = L.matmul(score, v)
    out = L.transpose(out, [0, 2, 1, 3])
    out = L.reshape(out, [0, 0, out.shape[2] * out.shape[3]])

    out = self.o(out)
    return out, cache


AttentionLayer.forward = _attn_forward


def _ernie_block_stack_forward(self,
                               inputs,
                               attn_bias=None,
                               past_cache=None,
                               num_layers=12,
                               depth_mult=1.,
                               head_mask=None):

    if past_cache is not None:
        assert isinstance(
            past_cache, tuple
        ), 'unknown type of `past_cache`, expect tuple or list. got %s' % repr(
            type(past_cache))
        past_cache = list(zip(*past_cache))
    else:
        past_cache = [None] * len(self.block)
    cache_list_k, cache_list_v, hidden_list = [], [], [inputs]

    depth = round(num_layers * depth_mult)
    kept_layers_index = []
    for i in range(1, depth + 1):
        kept_layers_index.append(math.floor(i / depth_mult) - 1)

    for i in kept_layers_index:
        b = self.block[i]
        p = past_cache[i]
        inputs, cache = b(inputs,
                          attn_bias=attn_bias,
                          past_cache=p,
                          head_mask=head_mask[i])

        cache_k, cache_v = cache
        cache_list_k.append(cache_k)
        cache_list_v.append(cache_v)
        hidden_list.append(inputs)

    return inputs, hidden_list, (cache_list_k, cache_list_v)


ErnieEncoderStack.forward = _ernie_block_stack_forward


def _ernie_block_forward(self,
                         inputs,
                         attn_bias=None,
                         past_cache=None,
                         head_mask=None):
    attn_out, cache = self.attn(
        inputs,
        inputs,
        inputs,
        attn_bias,
        past_cache=past_cache,
        head_mask=head_mask)  #self attn
    attn_out = self.dropout(attn_out)
    hidden = attn_out + inputs
    hidden = self.ln1(hidden)  # dropout/ add/ norm

    ffn_out = self.ffn(hidden)
    ffn_out = self.dropout(ffn_out)
    hidden = ffn_out + hidden
    hidden = self.ln2(hidden)
    return hidden, cache


ErnieBlock.forward = _ernie_block_forward


def _ernie_model_forward(self,
                         src_ids,
                         sent_ids=None,
                         pos_ids=None,
                         input_mask=None,
                         attn_bias=None,
                         past_cache=None,
                         use_causal_mask=False,
                         num_layers=12,
                         depth=1.,
                         head_mask=None):
    assert len(src_ids.shape
               ) == 2, 'expect src_ids.shape = [batch, sequecen], got %s' % (
                   repr(src_ids.shape))
    assert attn_bias is not None if past_cache else True, 'if `past_cache` is specified; attn_bias should not be None'
    d_batch = L.shape(src_ids)[0]
    d_seqlen = L.shape(src_ids)[1]
    if pos_ids is None:
        pos_ids = L.reshape(L.range(0, d_seqlen, 1, dtype='int32'), [1, -1])
        pos_ids = L.cast(pos_ids, 'int64')
    if attn_bias is None:
        if input_mask is None:
            input_mask = L.cast(src_ids != 0, 'float32')
        assert len(input_mask.shape) == 2
        input_mask = L.unsqueeze(input_mask, axes=[-1])
        attn_bias = L.matmul(input_mask, input_mask, transpose_y=True)
        if use_causal_mask:
            sequence = L.reshape(
                L.range(
                    0, d_seqlen, 1, dtype='float32') + 1., [1, 1, -1, 1])
            causal_mask = L.cast(
                (L.matmul(
                    sequence, 1. / sequence, transpose_y=True) >= 1.),
                'float32')
            attn_bias *= causal_mask
    else:
        assert len(
            attn_bias.shape
        ) == 3, 'expect attn_bias tobe rank 3, got %r' % attn_bias.shape
    attn_bias = (1. - attn_bias) * -10000.0
    attn_bias = L.unsqueeze(attn_bias, [1])
    attn_bias.stop_gradient = True

    if sent_ids is None:
        sent_ids = L.zeros_like(src_ids)

    if head_mask is not None:
        if len(head_mask.shape) == 1:
            head_mask = L.unsqueeze(
                L.unsqueeze(L.unsqueeze(L.unsqueeze(head_mask, 0), 0), -1), -1)
            head_mask = L.expand(
                head_mask, expand_times=[num_layers, 1, 1, 1, 1])
        elif len(head_mask.shape) == 2:
            head_mask = L.unsqueeze(
                L.unsqueeze(L.unsqueeze(head_mask, 1), -1), -1)

    else:
        head_mask = [None] * num_layers

    src_embedded = self.word_emb(src_ids)
    pos_embedded = self.pos_emb(pos_ids)
    sent_embedded = self.sent_emb(sent_ids)
    embedded = src_embedded + pos_embedded + sent_embedded

    embedded = self.dropout(self.ln(embedded))

    encoded, hidden_list, cache_list = self.encoder_stack(
        embedded,
        attn_bias,
        past_cache=past_cache,
        num_layers=num_layers,
        depth_mult=depth,
        head_mask=head_mask)
    if self.pooler is not None:
        pooled = self.pooler(encoded[:, 0, :])
    else:
        pooled = None

    additional_info = {
        'hiddens': hidden_list,
        'caches': cache_list,
    }

    if self.return_additional_info:
        return pooled, encoded, additional_info
    else:
        return pooled, encoded


ErnieModel.forward = _ernie_model_forward


def _seqence_forward(self, *args, **kwargs):
    labels = kwargs.pop('labels', None)
    pooled, encoded, additional_info = super(
        ErnieModelForSequenceClassification, self).forward(*args, **kwargs)
    hidden = self.dropout(pooled)
    logits = self.classifier(hidden)

    if labels is not None:
        if len(labels.shape) == 1:
            labels = L.reshape(labels, [-1, 1])
        loss = L.softmax_with_cross_entropy(logits, labels)
        loss = L.reduce_mean(loss)
    else:
        loss = None
    return loss, logits, additional_info


ErnieModelForSequenceClassification.forward = _seqence_forward


def get_config(pretrain_dir_or_url):
    bce = 'https://ernie-github.cdn.bcebos.com/'
    resource_map = {
        'ernie-1.0': bce + 'model-ernie1.0.1.tar.gz',
        'ernie-2.0-en': bce + 'model-ernie2.0-en.1.tar.gz',
        'ernie-2.0-large-en': bce + 'model-ernie2.0-large-en.1.tar.gz',
        'ernie-tiny': bce + 'model-ernie_tiny.1.tar.gz',
    }

    if not Path(pretrain_dir_or_url).exists() and str(
            pretrain_dir_or_url) in resource_map:
        url = resource_map[pretrain_dir_or_url]
        pretrain_dir = _fetch_from_remote(url, False)
    else:
        log.info('pretrain dir %s not in %s, read from local' %
                 (pretrain_dir_or_url, repr(resource_map)))
        pretrain_dir = Path(pretrain_dir_or_url)

    config_path = os.path.join(pretrain_dir, 'ernie_config.json')
    if not os.path.exists(config_path):
        raise ValueError('config path not found: %s' % config_path)
    cfg_dict = dict(json.loads(open(config_path).read()))
    return cfg_dict
