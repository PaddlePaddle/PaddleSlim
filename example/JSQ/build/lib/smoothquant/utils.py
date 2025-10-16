import copy

import paddle
from smoothquant.fake_quant import W8A8Linear

############################## 相关utils函数，如下 ##############################

def dim2perm(ndim, dim0, dim1):
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm

def _Tensor_reshape(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        assert "shape" in kwargs
        return paddle.reshape(self, shape=kwargs["shape"])

setattr(paddle.Tensor, "reshape", _Tensor_reshape)
############################## 相关utils函数，如上 ##############################



def clip_matrix(matrix, abs=True, clip_l=0, clip_h=0, channel=False):
    if clip_l == 0 and clip_h == 0:
        return matrix
    if channel:
        matrix_flatten = matrix
        if abs:
            matrix_flatten = paddle.abs(x=matrix)
        max_threshold = None
        min_threshold = None
        if clip_h != 0:
            max_threshold = paddle.quantile(
                x=matrix_flatten[0].astype(dtype="float64"), q=1 - clip_h, axis=0
            )
        clipped_matrix = paddle.clip(x=matrix, min=-max_threshold, max=max_threshold)
        return clipped_matrix
    else:
        num_elements = matrix.size
        if abs:
            matrix_flatten = paddle.abs(x=matrix).flatten()
        else:
            matrix_flatten = matrix.flatten()
        max_threshold = None
        min_threshold = None
        if clip_l != 0:
            low_index = int(clip_l * num_elements)
            min_threshold, _ = paddle.topk(x=matrix_flatten, largest=False, k=low_index)
            min_threshold = min_threshold[-1]
        if clip_h != 0:
            high_index = int(clip_h * num_elements)
            max_threshold, _ = paddle.topk(x=matrix_flatten, largest=True, k=high_index)
            max_threshold = max_threshold[-1]
        if abs:
            clipped_matrix = paddle.clip(
                x=matrix, min=-max_threshold, max=max_threshold
            )
        else:
            clipped_matrix = paddle.clip(x=matrix, min=min_threshold, max=max_threshold)
        return clipped_matrix


def find_layers(module, layers=[paddle.nn.Linear, W8A8Linear], name=""):
    if type(module) in layers or "FalconLinear" in module.__class__.__name__:
        return {name: module}
    else:
        pass
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def check_sparsity(model):
    CHATGLM = False
    Falcon = False
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "embedding"):
            CHATGLM = True
        elif hasattr(model.transformer, "word_embeddings"):
            Falcon = True
    if CHATGLM:
        layers = model.transformer.encoder.layers
    elif Falcon:
        layers = model.transformer.h
    else:
        layers = model.llama.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.size
            sub_count += (W == 0).sum().item()
            sub_params += W.size
        print(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")
    return float(count) / total_params


def prepare_calibration_input(model, dataloader, device):
    CHATGLM = False
    Falcon = False
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "embedding"):
            CHATGLM = True
        elif hasattr(model.transformer, "word_embeddings"):
            Falcon = True
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if CHATGLM:
        layers = model.transformer.encoder.layers
    elif Falcon:
        layers = model.transformer.h
    else:
        layers = model.llama.layers
    # if "model.embed_tokens" in model.hf_device_map:
    #     device = model.hf_device_map["model.embed_tokens"]
    device = model.llama.embed_tokens.weight.place
    dtype = next(iter(model.parameters())).dtype
    inps = paddle.zeros(
        shape=(128, model.seqlen, model.config.hidden_size), dtype=dtype
    )
    inps.stop_gradient = not False
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(paddle.nn.Layer):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, *args, **kwargs):
            if CHATGLM:
                inps[cache["i"]] = inp.transpose(perm=dim2perm(inp.ndim, 0, 1))[0]
            else:
                inps[cache["i"]] = inp
            cache["i"] += 1
            if CHATGLM:
                cache["attention_mask"] = args[0]
            else:
                # cache["attention_mask"] = kwargs["attention_mask"]
                cache["attention_mask"] = kwargs.get("attention_mask", None)
            if CHATGLM:
                cache["position_ids"] = args[1]
            elif Falcon:
                pass
            else:
                # cache["position_ids"] = kwargs["position_ids"]
                cache["position_ids"] = kwargs.get("position_ids", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            if cache["i"] == 0:
                model_inputs = model.prepare_inputs_for_generation(batch[0].to(device))
                if "position_ids" in model_inputs:
                    cache["position_ids"] = model_inputs["position_ids"]
                else:
                    cache["position_ids"] = paddle.arange(
                        start=0, end=tuple(batch[0].shape)[1], dtype="int64"
                    ).unsqueeze(axis=0)
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    outs = paddle.zeros_like(x=inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    model.config.use_cache = use_cache
    return inps, outs, attention_mask, position_ids


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = paddle.take_along_axis(
        arr=sort_res[0],
        axis=1,
        indices=sort_mask.sum(dim=1, keepdims=True) - 1,
        broadcast=False,
    )
    W_mask = W_metric <= thres
    cur_sparsity = (W_mask == True).sum() / W_mask.size
    return W_mask, cur_sparsity


def cal_mse_layer(
    args, layer1, layer2, inps, attention_mask, position_ids, outs1=None, outs2=None
):
    if outs1 is None:
        layer1_outs = []
        for i in range(args.nsamples):
            with paddle.no_grad():
                layer1_outs.append(
                    layer1(
                        inps[i].unsqueeze(axis=0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0][0]
                )
    else:
        layer1_outs = outs1
    if outs2 is None:
        layer2_outs = []
        for i in range(args.nsamples):
            with paddle.no_grad():
                layer2_outs.append(
                    layer2(
                        inps[i].unsqueeze(axis=0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0][0]
                )
    else:
        layer2_outs = outs2
    mse = 0.0
    print(tuple(layer1_outs[0].shape))
    for i in range(args.nsamples):
        device = layer2_outs[i].place
        mse += paddle.nn.functional.mse_loss(
            input=layer1_outs[i].to(device), label=layer2_outs[i]
        ).item()
    return mse


def generate_ss(activation, weight):
    # weight: (out_features, in_features) for Paddle
    cout, cin = weight.shape
    ss = paddle.zeros_like(weight)
    
    # 确保 activation 是 2D
    if len(activation.shape) == 1:
        activation = activation.unsqueeze(0)
    for i in range(cout):
        w = weight.clone()
        w[i, :] = 0
        
        # w: (cout, cin), 需要转置为 (cin, cout)
        # activation: (batch, cin) @ w.T(cin, cout) = (batch, cout)
        # w_t = w.transpose([1, 0])  # 显式转置
        out = paddle.matmul(activation, w)
        
        max_values = paddle.max(out, axis=0)
        min_values = paddle.min(out, axis=0)
        
        ss[i, :] = max_values - min_values
    
    ss = paddle.where(
        paddle.isinf(ss), 
        paddle.to_tensor(100.0, dtype=ss.dtype), 
        ss
    )
    ss = ss.transpose([1, 0])
    return ss


def generate_ss2(activation, weight):
    out = activation @ weight.t()
    max_values, _ = paddle.max(x=out, axis=0), paddle.argmax(x=out, axis=0)
    min_values, _ = paddle.min(x=out, axis=0), paddle.argmin(x=out, axis=0)
    row_ss = (max_values - min_values).reshape((-1, 1))
    return row_ss