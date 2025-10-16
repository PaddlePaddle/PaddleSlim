import math
import os
import pdb
import random
import time
from copy import deepcopy
from paddlenlp.datasets import load_dataset
import numpy as np
import paddle
import paddlenlp
from loguru import logger
from smoothquant.layerwrapper import WrappedGPT
from smoothquant.quantize import quantize_layer
from smoothquant.smooth import smooth_layer
from smoothquant.utils import *

# ############################# 相关utils函数，如下 ##############################
def device2str(type=None, index=None, *, device=None):
    type = device if device else type
    if isinstance(type, int):
        type = f'gpu:{type}'
    elif isinstance(type, str):
        if 'cuda' in type:
            type = type.replace('cuda', 'gpu')
        if 'cpu' in type:
            type = 'cpu'
        elif index is not None:
            type = f'{type}:{index}'
    elif isinstance(type, paddle.CPUPlace) or (type is None):
        type = 'cpu'
    elif isinstance(type, paddle.CUDAPlace):
        type = f'gpu:{type.get_device_id()}'

    return type

def _Tensor_view(self, *args, **kwargs):
    if args:
        if len(args)==1 and isinstance(args[0], (tuple, list, str)):
            return paddle.view(self, args[0])
        else:
            return paddle.view(self, list(args))
    elif kwargs:
        return paddle.view(self, shape_or_dtype = list(kwargs.values())[0])

setattr(paddle.Tensor, 'view', _Tensor_view)

def paddle_max(*args, **kwargs):
    if "input" in kwargs:
        kwargs["x"] = kwargs.pop("input")

    out_v = None
    if "out" in kwargs:
        out_v = kwargs.pop("out")

    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(*args, **kwargs)
    elif len(args)==2 and isinstance(args[1], paddle.Tensor):
        ret = paddle.maximum(*args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 2:
            if out_v:
                ret = paddle.max(*args, **kwargs), paddle.argmax(*args, **kwargs)
                paddle.assign(ret[0], out_v[0])
                paddle.assign(ret[1], out_v[1])
                return out_v
            else:
                ret = paddle.max(*args, **kwargs), paddle.argmax(*args, **kwargs)
                return ret
        else:
            ret = paddle.max(*args, **kwargs)
            return ret

    if out_v:
        paddle.assign(ret, out_v)
        return out_v
    else:
        return ret
############################## 相关utils函数，如上 ##############################



global layer_num
layer_num = 0


def prune_wanda_and_smoothquant_annealing(args, device=device2str("cuda:0")):
    logger.info(f"test_only is {args.test_only}")
    logger.info(f"rho is {args.rho}")
    logger.info(f"sparsity_ratio is {args.sparsity_ratio}")
    logger.info(f"nbits is {args.nbits}")
    model = build_model(args)
    CHATGLM = False
    if CHATGLM:
        logger.info("chatglm")
        exit(0)
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "embedding"):
            CHATGLM = True
            logger.info("chatglm")
    logger.info(f"sparsity type: {args.sparsity_type}")
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert (
            args.sparsity_ratio == 0.5
        ), "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
    logger.info(f"set n to {prune_n}, set m to {prune_m}")
    # T = 300
    # Tmin = 10
    # k = 5
    # t = 0
    if CHATGLM:
        layers = model.transformer.encoder.layers
    else:
        layers = model.llama.layers
    clip_opts = [0.0, 4e-05, 5e-05, 6e-05, 7e-05]
    clip_table = [2] * len(layers)
    # ppl_prev = clip(model, args, clip_opts, clip_table, device, prune_n, prune_m)
    clip(model, args, clip_opts, clip_table, device, prune_n, prune_m)
    model.save_pretrained(args.saved_path)
    tokenizer.save_pretrained(args.saved_path)
    logger.info(f"model saved to {args.saved_path}")
    # ppl_new = ppl_prev
    # best_ppl = ppl_new
    # if args.test_only:
    #     logger.info(f"ppl is {best_ppl}")
    #     model.save_pretrained(args.saved_path)
    #     exit(0)
    # while T >= Tmin:
    #     for i in range(k):
    #         random.seed(time.time())
    #         change_axis = random.randint(0, len(layers) - 1)
    #         new_clip_table = deepcopy(clip_table)
    #         change_opt = random.randint(0, 4)
    #         while change_opt == new_clip_table[change_axis]:
    #             change_opt = random.randint(0, 4)
    #         new_clip_table[change_axis] = change_opt
    #         logger.info(f"new table is {new_clip_table}")
    #         new_model = build_model(args)
    #         ppl_new = clip(
    #             new_model, args, clip_opts, new_clip_table, device, prune_n, prune_m
    #         )
    #         if ppl_new - ppl_prev < 0:
    #             logger.info(
    #                 f"ppl_new is {ppl_new}, ppl_prev is {ppl_prev}, ppl_new is better, transfer"
    #             )
    #             clip_table = new_clip_table
    #             ppl_prev = ppl_new
    #         else:
    #             logger.info(
    #                 f"ppl_new is {ppl_new}, ppl_prev is {ppl_prev}, ppl_prev is better, do metropolis"
    #             )
    #             p = math.exp(-(ppl_new - ppl_prev) * 1500 / T)
    #             r = np.random.uniform(low=0, high=1)
    #             if r <= p:
    #                 logger.info(f"r is {r}, p is {p}, r <= p. transfer")
    #                 clip_table = new_clip_table
    #                 ppl_prev = ppl_new
    #             else:
    #                 logger.info(f"r is {r}, p is {p}, r > p. do not transfer")
    #         if best_ppl > ppl_prev:
    #             best_ppl = ppl_prev
    #             logger.info(
    #                 f"best ppl is {best_ppl}, now ppl is {ppl_prev}, store now model."
    #             )
    #             new_model.save_pretrained(args.saved_path)
    #         else:
    #             logger.info(
    #                 f"best ppl is {best_ppl}, now ppl is {ppl_prev}, next model."
    #             )
    #         logger.info(
    #             f"now T is {T}, epoch is {i}, now ppl is {ppl_prev}, now clip table is {clip_table}"
    #         )
    #     t += 1
    #     T = T / (1 + t)
    paddle.device.cuda.empty_cache()


def clip(model, args, clip_opts, clip_table, device, prune_n=0, prune_m=0):
    logger.info("loading calibdation data")
    dataloader, testenc = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )
    logger.info("dataset loading complete")
    with paddle.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device
        )
    CHATGLM = False
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "embedding"):
            CHATGLM = True
    if CHATGLM:
        layers = model.transformer.encoder.layers
    else:
        layers = model.llama.layers
    for i in range(len(layers)):
        logger.info(f"layer is {i}")
        layer = layers[i]
        layer_name = f"model.layers.{i}"
        subset = find_layers(layer)
        # import pdb; pdb.set_trace()
        if (
            any(s in args.model for s in ["30b", "70b"])
            and "model.layers.{i}" in model.hf_device_map
        ):
            logger.info("multi-dev")
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
        act_scales = {}

        def stat_tensor(name, tensor):
            hidden_dim = tuple(tensor.shape)[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = (
                (paddle.max(x=tensor, axis=0), paddle.argmax(x=tensor, axis=0))[0]
                .astype(dtype="float32")
                .cpu()
            )
            if name in act_scales:
                act_scales[layer_name + "." + name] = paddle_max(
                    act_scales[name], comming_max
                )
            else:
                act_scales[layer_name + "." + name] = comming_max

        def add_batch(name):
            def tmp(_, inp, out):
                inp = inp[0].data
                inp = clip_matrix(inp, args.abs, args.clip_l, clip_opts[clip_table[i]])
                stat_tensor(name, inp)
                wrapped_layers[name].add_batch(inp, out.data)

            return tmp

        handles = []
        logger.info(f"nsamples is {args.nsamples}")
        for name in wrapped_layers:
            handles.append(
                subset[name].register_forward_post_hook(hook=add_batch(name))
            )
        for j in range(args.nsamples):
            with paddle.no_grad():
                if CHATGLM:
                    outs[j] = layer(
                        inps[j].unsqueeze(axis=0), attention_mask, position_ids
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(axis=0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        for h in handles:
            h.remove()
        for name in subset:
            weight = paddle.abs(subset[name].weight.data)
            activation = paddle.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            ss = generate_ss(
                wrapped_layers[name].inp_sum / wrapped_layers[name].inp_num,
                subset[name].weight.data,
            )
            # if weight.shape[0] != weight.shape[1]:
            #     import pdb; pdb.set_trace()
            W_metric = weight.transpose([1,0]) * activation + args.rho * ss

            # 初始化为 float32，最后转换为 bool
            W_mask = paddle.zeros_like(W_metric, dtype='float32')

            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : ii + prune_m].astype(dtype="float32")
                        topk_indices = ii + paddle.topk(tmp, k=prune_n, axis=1, largest=False)[1]
                        # 使用 float32 进行 put_along_axis 操作
                        W_mask = paddle.put_along_axis(
                            W_mask,
                            topk_indices,
                            paddle.ones_like(topk_indices, dtype='float32'),
                            axis=1
                        )
            else:
                # unstructured pruning
                sorted_indices = paddle.argsort(W_metric, axis=-1, stable=True)
                indices = sorted_indices[:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                # 使用 float32 进行 put_along_axis 操作
                W_mask = paddle.put_along_axis(
                    W_mask,
                    indices,
                    paddle.ones_like(indices, dtype='float32'),
                    axis=1
                )

            # 转换为 bool 类型
            W_mask = W_mask.astype('bool')

            # 将 mask 位置的权重置零
            subset[name].weight.set_value(
                paddle.where(W_mask.transpose([1,0]), paddle.zeros_like(subset[name].weight), subset[name].weight)
            )
            
        for j in range(args.nsamples):
            with paddle.no_grad():
                if CHATGLM:
                    outs[j] = layer(
                        inps[j].unsqueeze(axis=0), attention_mask, position_ids
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(axis=0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
        smooth_layer(layer_name, layer, act_scales, 0.8)
        quantize_layer(layer, nbits=args.nbits)
        inps, outs = outs, inps
    # logger.info("begin eval")
    ppl = 0
    # if CHATGLM:
    #     ppl = chatglm_eval(model, testenc, device)
    # else:
    #     ppl = llama_eval(model, testenc, device)
    # logger.info(f"SmoothQuant W8A8 quantized model ppl: {ppl}")
    # model.cpu()
    paddle.device.cuda.empty_cache()
    return ppl


@paddle.no_grad()
def chatglm_eval(model, testenc, dev):
    model.eval()
    testenc = testenc.input_ids
    nsamples = testenc.size // model.seqlen
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.encoder.layers
    model.transformer.embedding.word_embeddings = (
        model.transformer.embedding.word_embeddings.to(dev)
    )
    model.transformer.rotary_pos_emb = model.transformer.rotary_pos_emb.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = paddle.zeros(
        shape=(nsamples, model.seqlen, model.config.hidden_size), dtype=dtype
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(paddle.nn.Layer):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(
            self,
            inp,
            attention_mask=None,
            position_ids=None,
            kv_cache=None,
            use_cache=True,
        ):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = attention_mask
            cache["position_ids"] = position_ids
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, i * model.seqlen : (i + 1) * model.seqlen].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.transformer.embedding.word_embeddings = (
        model.transformer.embedding.word_embeddings.cpu()
    )
    paddle.device.cuda.empty_cache()
    outs = paddle.zeros_like(x=inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    for i in range(len(layers)):
        if (
            "30b" in model.config._name_or_path.lower()
            and "model.layers.{i}" in model.hf_device_map
        ):
            dev = model.hf_device_map[f"model.layers.{i}"]
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(axis=0), attention_mask, position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        paddle.device.cuda.empty_cache()
        inps, outs = outs, inps
    if model.transformer.encoder.final_layernorm is not None:
        model.transformer.encoder.final_layernorm = (
            model.transformer.encoder.final_layernorm.to(dev)
        )
    model.transformer.output_layer = model.transformer.output_layer.to(dev)
    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(axis=0)
        if model.transformer.encoder.final_layernorm is not None:
            hidden_states = model.transformer.encoder.final_layernorm(hidden_states)
        lm_logits = model.transformer.output_layer(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, i * model.seqlen : (i + 1) * model.seqlen][:, 1:]
        loss_fct = paddle.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.astype(dtype="float32") * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = paddle.exp(x=paddle.stack(x=nlls).sum() / (nsamples * model.seqlen))
    model.config.use_cache = use_cache
    return ppl


@paddle.no_grad()
def llama_eval(model, testenc, dev):
    model.eval()
    testenc = testenc.input_ids
    nsamples = testenc.size // model.seqlen
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.llama.layers
    model.llama.embed_tokens = model.llama.embed_tokens.to(dev) #改前为 model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = paddle.zeros(
        shape=(nsamples, model.seqlen, model.config.hidden_size), dtype=dtype
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(paddle.nn.Layer):
        def __init__(self, module):
            super().__init__()
            self.module = module

        # 改之前
        # def forward(self, inp, **kwargs):
        #     inps[cache["i"]] = inp
        #     cache["i"] += 1
        #     cache["attention_mask"] = kwargs["attention_mask"]
        #     cache["position_ids"] = kwargs["position_ids"]
        #     raise ValueError

        def forward(self, *args, **kwargs):
            inp = args[0]
            inps[cache["i"]] = inp
            cache["i"] += 1
            # Paddle 的 LlamaDecoderLayer 参数顺序为：
            # hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position
            if len(args) > 1:
                cache["attention_mask"] = args[1]
            if len(args) > 2:
                cache["position_ids"] = args[2]

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = paddle.to_tensor(testenc[:, i * model.seqlen : (i + 1) * model.seqlen], dtype='int64', place=dev) 
        #改之前 batch = testenc[:, i * model.seqlen : (i + 1) * model.seqlen].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.llama.embed_tokens = model.llama.embed_tokens.cpu() #改之前model.model.embed_tokens = model.model.embed_tokens.cpu()
    paddle.device.cuda.empty_cache()
    outs = paddle.zeros_like(x=inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    for i in range(len(layers)):
        if (
            any(s in model.config._name_or_path.lower() for s in ["30b", "70b"])
            and "model.layers.{i}" in model.hf_device_map
        ):
            logger.info("multi-dev")
            dev = model.hf_device_map[f"model.layers.{i}"]
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(axis=0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
        layers[i] = layer.cpu()
        del layer
        paddle.device.cuda.empty_cache()
        inps, outs = outs, inps
    if model.llama.norm is not None:
        model.llama.norm = model.llama.norm.to(dev)
    # 改之前if model.model.norm is not None:
    #     model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)
    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(axis=0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, i * model.seqlen : (i + 1) * model.seqlen][:, 1:]
        loss_fct = paddle.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.astype(dtype="float32") * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = paddle.exp(x=paddle.stack(x=nlls).sum() / (nsamples * model.seqlen))
    model.config.use_cache = use_cache
    return ppl


def get_wikitext2(nsamples, seed, seqlen, model):
    # wufazidong
    import datasets
    # traindata = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # testdata = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # traindata = load_dataset('wikitext', splits='train')
    # testdata = load_dataset('wikitext', splits='test')
    # traindata = load_dataset("glue", "sst-2", splits="train")
    # testdata = load_dataset("glue", "sst-2", splits="test")
    traindata, devdata, testdata = load_dataset("ptb")
    if "glm" in model:
        # wufazidong
        # tokenizer = transformers.AutoTokenizer.from_pretrained(
        #     model, trust_remote_code=True, use_fast=False
        # )
        tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained(
            model, trust_remote_code=True, use_fast=False
        )
    else:
        # tokenizer = transformers.LlamaTokenizer.from_pretrained(model, use_fast=False)
        tokenizer = paddlenlp.transformers.LlamaTokenizer.from_pretrained(model, use_fast=False)
    # trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
    train_text = "\n\n".join([example["sentence"] for example in traindata])
    trainenc = tokenizer(train_text, return_tensors="np")
    # testenc = tokenizer("\n\n".join(testdata["sentence"]), return_tensors="pt")
    test_text = "\n\n".join([example["sentence"] for example in testdata])
    testenc = tokenizer(test_text, return_tensors="np")
    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        # tar = inp.clone()
        inp = paddle.to_tensor(inp)   # 把 numpy 转成 Paddle Tensor
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model):
    logger.info("load c4 datasets")
    logger.info("load from local")
    # wufazidong
    # traindata = datasets.load_from_disk(
    #     "/mnt/nvme0/wangzining/hf/allenai/c4/allenai--c4/train"
    # )
    traindata = paddlenlp.datasets.load_from_disk(
        "/mnt/nvme0/wangzining/hf/allenai/c4/allenai--c4/train"
    )
    # wufazidong
    # valdata = datasets.load_from_disk(
    #     "/mnt/nvme0/wangzining/hf/allenai/c4/allenai--c4/validation"
    # )
    valdata = paddlenlp.datasets.load_from_disk(
        "/mnt/nvme0/wangzining/hf/allenai/c4/allenai--c4/validation"
    )
    if "glm" in model:
        # wufazidong
        # tokenizer = transformers.AutoTokenizer.from_pretrained(
        #     model, trust_remote_code=True, use_fast=False
        # )
        tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained(
            model, use_auth_token=None  # trust_remote_code 在 PaddleNLP 中没有直接对应参数，但可以通过设置 use_auth_token=None 来避免加载远程代码时的认证问题（如果不需要的话）
        )
    else:
        # wufazidong
        # tokenizer = transformers.LlamaTokenizer.from_pretrained(model, use_fast=False)
        tokenizer = paddlenlp.transformers.LlamaTokenizer.from_pretrained(model, use_fast=False)
    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    import random

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = paddle.hstack(x=valenc)

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_llama(model):
    def skip(*args, **kwargs):
        pass

    paddle.nn.initializer.KaimingUniform = skip
    paddle.nn.initializer.Uniform = skip
    paddle.nn.initializer.Normal = skip
    kwargs = {"torch_dtype": "float16", "device_map": "auto"}
    # wufazidong
    # model = transformers.AutoModelForCausalLM.from_pretrained(model, **kwargs)
    model = paddlenlp.transformers.AutoModelForCausalLM.from_pretrained(model, **kwargs)
    model.seqlen = 2048
    return model


def build_model(args):
    model_name = args.model
    if "glm" in model_name:
        kwargs = {
            "torch_dtype": "float16",
            "device_map": "auto",
            "trust_remote_code": True,
        }
    else:
        kwargs = {
            # "torch_dtype": "float16", # caozuo1:注释掉，否则会报错。paddlenlp 不支持 torch_dtype（这是 HF Transformers 的参数
            # "device_map": "auto" # caozuo1:注释掉，否则会报错。paddlenlp 不支持 device_map（这是 HF Transformers 的参数
            }
    # wufazidong
    # model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model = paddlenlp.transformers.AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.seqlen = args.seqlen
    return model


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=""):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, model)