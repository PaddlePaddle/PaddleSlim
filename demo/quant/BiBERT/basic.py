import collections
from webbrowser import get

import paddle
from paddle import tensor
from paddle.autograd import PyLayer
from paddle.nn import functional as F
from paddle.nn.layer.common import Linear, Embedding
from paddle.nn.layer.transformer import MultiHeadAttention, _convert_attention_mask


class BinaryQuantizer(PyLayer):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = paddle.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensor()[0]
        grad_input = grad_output
        grad_input[input >= 1] = 0
        grad_input[input <= -1] = 0
        return grad_input.clone()


class ZMeanBinaryQuantizer(PyLayer):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = (paddle.sign(input) + 1) / 2
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensor()[0]
        grad_input = grad_output
        grad_input[input >= 1] = 0
        grad_input[input <= -1] = 0
        return grad_input.clone()


class BiLinear(Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 weight_attr=None,
                 bias_attr=None,
                 name=None):
        super(BiLinear, self).__init__(
            in_features,
            out_features,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            name=name)

    def forward(self, input):
        scaling_factor = paddle.mean(
            self.weight.abs(), axis=1).unsqueeze(1).detach()
        real_weights = self.weight - paddle.mean(
            self.weight, axis=-1).unsqueeze(-1)
        binary_weights_no_grad = scaling_factor * paddle.sign(real_weights)
        cliped_weights = paddle.clip(real_weights, -1.0, 1.0)
        weight = binary_weights_no_grad.detach() - cliped_weights.detach(
        ) + cliped_weights

        binary_input_no_grad = paddle.sign(input)
        cliped_input = paddle.clip(input, -1.0, 1.0)
        ba = binary_input_no_grad.detach() - cliped_input.detach(
        ) + cliped_input

        out = F.linear(x=ba, weight=weight, bias=self.bias, name=self.name)
        return out


class BiEmbedding(Embedding):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 sparse=False,
                 weight_attr=None,
                 name=None):
        super(BiEmbedding,
              self).__init__(num_embeddings, embedding_dim, padding_idx, sparse,
                             weight_attr, name)

    def forward(self, x):
        scaling_factor = paddle.mean(self.weight.abs(), axis=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        real_weights = self.weight - paddle.mean(
            self.weight, axis=-1, keepdim=True)
        binary_weights_no_grad = scaling_factor * paddle.sign(real_weights)
        cliped_weights = paddle.clip(real_weights, -1.0, 1.0)
        weight = binary_weights_no_grad.detach() - cliped_weights.detach(
        ) + cliped_weights
        return F.embedding(
            x,
            weight=weight,
            padding_idx=self._padding_idx,
            sparse=self._sparse,
            name=self._name)


class BiMultiHeadAttention(MultiHeadAttention):
    # fork from paddle.nn.layer.transformer.MultiHeadAttention
    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False,
                 weight_attr=None,
                 bias_attr=None):
        super(BiMultiHeadAttention,
              self).__init__(embed_dim, num_heads, dropout, kdim, vdim,
                             need_weights, weight_attr, bias_attr)

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if cache is None:
            q, k, v = self._prepare_qkv(query, key, value, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, cache)

        q = BinaryQuantizer.apply(q)
        k = BinaryQuantizer.apply(k)

        product = paddle.matmul(x=q, y=k, transpose_y=True)
        product = paddle.scale(product, scale=self.head_dim**-0.5)
        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")
        weights = ZMeanBinaryQuantizer.apply(weights)
        v = BinaryQuantizer.apply(v)

        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


def _to_bi_function(model):
    for name, layer in model.named_children():
        if isinstance(layer, MultiHeadAttention):
            new_layer = BiMultiHeadAttention(
                layer.embed_dim, layer.num_heads, layer.dropout, layer.kdim,
                layer.vdim, layer.need_weights, layer.q_proj._weight_attr,
                layer.q_proj._bias_attr)
            new_layer.q_proj = layer.q_proj
            new_layer.k_proj = layer.k_proj
            new_layer.v_proj = layer.v_proj
            new_layer.out_proj = layer.out_proj
            model._sub_layers[name] = new_layer
        elif isinstance(layer, Embedding):
            if name != "word_embeddings": continue
            new_layer = BiEmbedding(layer._num_embeddings, layer._embedding_dim,
                                    layer._padding_idx, layer._sparse,
                                    layer._weight_attr, layer._name)
            new_layer.weight = layer.weight
            model._sub_layers[name] = new_layer
        elif isinstance(layer, Linear):
            if name == "classifier": continue
            new_layer = BiLinear(layer.weight.shape[0], layer.weight.shape[1],
                                 layer._weight_attr, layer._bias_attr,
                                 layer.name)
            new_layer.weight = layer.weight
            new_layer.bias = layer.bias
            model._sub_layers[name] = new_layer


import math


def _MultiHeadAttention_forward(self,
                                query,
                                key=None,
                                value=None,
                                attn_mask=None,
                                cache=None):
    key = query if key is None else key
    value = query if value is None else value
    # compute q ,k ,v
    if cache is None:
        q, k, v = self._prepare_qkv(query, key, value, cache)
    else:
        q, k, v, cache = self._prepare_qkv(query, key, value, cache)

    # distill qxq
    query_scores = paddle.matmul(q, tensor.transpose(x=q, perm=[0, 1, 3, 2]))
    query_scores = query_scores / math.sqrt(self.head_dim)
    # distill kxk
    key_scores = paddle.matmul(k, tensor.transpose(x=k, perm=[0, 1, 3, 2]))
    key_scores = key_scores / math.sqrt(self.head_dim)

    product = paddle.matmul(x=q, y=k, transpose_y=True)
    product = paddle.scale(product, scale=self.head_dim**-0.5)
    if attn_mask is not None:
        # Support bool or int mask
        attn_mask = _convert_attention_mask(attn_mask, product.dtype)
        product = product + attn_mask
    weights = F.softmax(product)
    if self.dropout:
        weights = F.dropout(
            weights,
            self.dropout,
            training=self.training,
            mode="upscale_in_train")

    # distil vxv
    value_scores = paddle.matmul(v, tensor.transpose(x=v, perm=[0, 1, 3, 2]))
    value_scores = value_scores / math.sqrt(self.head_dim)
    out = tensor.matmul(weights, v)

    # combine heads
    out = tensor.transpose(out, perm=[0, 2, 1, 3])
    out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

    # project to output
    out = self.out_proj(out)

    outs = [out]
    if self.need_weights:
        outs.append(weights)
    if cache is not None:
        outs.append(cache)

    self.query_scores = query_scores
    self.key_scores = key_scores
    self.value_scores = value_scores
    return out if len(outs) == 1 else tuple(outs)


def _Bi_MultiHeadAttention_forward(self,
                                   query,
                                   key=None,
                                   value=None,
                                   attn_mask=None,
                                   cache=None):
    key = query if key is None else key
    value = query if value is None else value
    # compute q ,k ,v
    if cache is None:
        q, k, v = self._prepare_qkv(query, key, value, cache)
    else:
        q, k, v, cache = self._prepare_qkv(query, key, value, cache)

    # distill qxq
    query_scores = paddle.matmul(q, tensor.transpose(x=q, perm=[0, 1, 3, 2]))
    query_scores = query_scores / math.sqrt(self.head_dim)
    # distill kxk
    key_scores = paddle.matmul(k, tensor.transpose(x=k, perm=[0, 1, 3, 2]))
    key_scores = key_scores / math.sqrt(self.head_dim)

    q = BinaryQuantizer.apply(q)
    k = BinaryQuantizer.apply(k)

    product = paddle.matmul(x=q, y=k, transpose_y=True)
    product = paddle.scale(product, scale=self.head_dim**-0.5)
    if attn_mask is not None:
        # Support bool or int mask
        attn_mask = _convert_attention_mask(attn_mask, product.dtype)
        product = product + attn_mask

#    weights = F.softmax(product)
    weights = product
    if self.dropout:
        weights = F.dropout(
            weights,
            self.dropout,
            training=self.training,
            mode="upscale_in_train")

    # distil vxv
    value_scores = paddle.matmul(v, tensor.transpose(x=v, perm=[0, 1, 3, 2]))
    value_scores = value_scores / math.sqrt(self.head_dim)

    weights = ZMeanBinaryQuantizer.apply(weights)
    v = BinaryQuantizer.apply(v)

    out = tensor.matmul(weights, v)

    # combine heads
    out = tensor.transpose(out, perm=[0, 2, 1, 3])
    out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

    # project to output
    out = self.out_proj(out)

    outs = [out]
    if self.need_weights:
        outs.append(weights)
    if cache is not None:
        outs.append(cache)

    self.query_scores = query_scores
    self.key_scores = key_scores
    self.value_scores = value_scores
    return out if len(outs) == 1 else tuple(outs)


def _TransformerEncoderLayer_forward(self, src, src_mask=None, cache=None):
    src_mask = _convert_attention_mask(src_mask, src.dtype)

    residual = src
    if self.normalize_before:
        src = self.norm1(src)
    # Add cache for encoder for the usage like UniLM
    if cache is None:
        src = self.self_attn(src, src, src, src_mask)
    else:
        src, incremental_cache = self.self_attn(src, src, src, src_mask, cache)

    src = residual + self.dropout1(src)
    if not self.normalize_before:
        src = self.norm1(src)

    residual = src
    if self.normalize_before:
        src = self.norm2(src)
    src = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = residual + self.dropout2(src)
    if not self.normalize_before:
        src = self.norm2(src)
    self.rep = src
    return src if cache is None else (src, incremental_cache)


def _get_attr(model, attr):
    res = []
    if hasattr(model, attr):
        res.append(getattr(model, attr))
    for layer in model.children():
        res.extend(_get_attr(layer, attr))
    return res


def _to_distill_function(model):
    from types import MethodType
    for layer in model.children():
        if isinstance(layer, BiMultiHeadAttention):
            layer.forward = MethodType(_Bi_MultiHeadAttention_forward, layer)
        elif isinstance(layer, MultiHeadAttention):
            layer.forward = MethodType(_MultiHeadAttention_forward, layer)
        elif isinstance(layer,
                        paddle.nn.layer.transformer.TransformerEncoderLayer):
            layer.forward = MethodType(_TransformerEncoderLayer_forward, layer)
