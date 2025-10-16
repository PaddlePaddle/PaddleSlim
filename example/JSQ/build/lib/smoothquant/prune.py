import copy

import torch
import torch.nn as nn
from smoothquant.smooth import smooth_layer
from smoothquant.quantize import quantize_layer
from smoothquant.data import get_loaders
from smoothquant.layerwrapper import WrappedGPT
from smoothquant.sparsegpt import SparseGPT
from smoothquant.utils import find_layers, prepare_calibration_input, return_given_alpha, cal_mse_layer, check_sparsity, \
    clip_matrix, generate_ss, generate_ss2
import functools
import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
)


def build_model_and_tokenizer(args, model_name):
    while True:
        try:
            if "llama" in model_name:
                tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
            break
        except Exception as e:
            print(f"Error: {e}")
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    while True:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs, cache_dir=args.cache_dir)
            model.seqlen = model.config.max_position_embeddings
            break
        except Exception as e:
            print(f"Error: {e}")
    return model, tokenizer
import sys
sys.path.append("..")
from orisparsegpt.llama import llama_sparsegpt
def joint_pq(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if not args.test_sparsegpt and not args.test_wanda and not args.test_quant:
        return model
    print(args.sparsity_ratio)
    print(args.nbits)
    print(args.clip_h)
    # return llama_sparsegpt(model, tokenizer)
    if args.test_sparsegpt:
        return prune_sparsegpt(args, model, tokenizer, dev=torch.device("cuda:0"), prune_n=0, prune_m=0)
    else:
        return prune_wanda_and_smoothquant(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0)

def prune_wanda_and_smoothquant(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    CHATGLM = False
    Falcon = False
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "embedding"):
            CHATGLM = True
        elif hasattr(model.transformer, "word_embeddings"):
            Falcon = True
    if not args.test_clip:
        args.clip_h = 0.0
    print(model)
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(model, dataloader, device)

    if CHATGLM:
        layers = model.transformer.encoder.layers
    elif Falcon:
        layers = model.transformer.h
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        layer_name = f'model.layers.{i}'
        # quantize_layer(layer)

        subset = find_layers(layer)
        # print(subset)
        if f"model.layers.{i}" in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(
                dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        print(f"get scales of layer {i} and pruning")
        act_scales = {}

        def stat_tensor(name, tensor):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()
            print(name)
            print(comming_max)
            if name in act_scales:
                act_scales[layer_name + '.' + name] = torch.max(act_scales[name], comming_max)
            else:
                act_scales[layer_name + '.' + name] = comming_max

        def add_batch(name):
            def tmp(_, inp, out):
                inp = clip_matrix(inp[0].data, args.abs, 0, args.clip_h)
                stat_tensor(name, inp)
                wrapped_layers[name].add_batch(inp, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if CHATGLM:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask, position_ids)[0]
                elif Falcon:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=None)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            weight = torch.abs(subset[name].weight.data)
            activation = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            if args.test_ss:
                ss = generate_ss(wrapped_layers[name].inp_sum / wrapped_layers[name].inp_num, subset[name].weight.data)


            # mm, _ = torch.max(ss, dim=1)
            # mm2, _ = torch.min(ss, dim=1)
            # print(mm - mm2)
            # print(torch.mean(ss))
            # print(torch.mean(weight * activation))

            # args.rho = torch.abs(torch.mean(weight * activation) / torch.mean(ss))
            # print(f"{args.sparsity_ratio} {args.clip_h} {args.rho}")
            if args.mul:
                W_metric = weight * activation# * ss
            else:
                W_metric = weight * activation# + args.rho * ss

            if args.test_ss:
                W_metric = W_metric + args.rho * ss

            W_mask = (torch.zeros_like(W_metric) == 1)
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                # unstructured pruning
                indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            if args.test_wanda:
                subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                if CHATGLM:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask, position_ids)[0]
                elif Falcon:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=None)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        if args.test_smooth:
            print(f"smoothing layer {i}")
            smooth_layer(layer_name, layer, act_scales, 0.8)


        if args.test_quant:
            print(f"quantizing layer {i}")
            quantize_layer(layer, nbits=args.nbits)

        inps, outs = outs, inps

    torch.cuda.empty_cache()
    print(model)
    return model


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev=torch.device("cuda:0"), prune_n=0, prune_m=0):
    CHATGLM = False
    Falcon = False
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "embedding"):
            CHATGLM = True
        elif hasattr(model.transformer, "word_embeddings"):
            Falcon = True

    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    if CHATGLM:
        layers = model.transformer.encoder.layers
    elif Falcon:
        layers = model.transformer.h
    else:
        layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            if CHATGLM:
                inps[cache['i']] = inp.transpose(0, 1)[0]
            else:
                inps[cache['i']] = inp
            cache['i'] += 1
            if CHATGLM:
                cache['attention_mask'] = args[0]
            else:
                cache['attention_mask'] = kwargs['attention_mask']
            if CHATGLM:
                cache['position_ids'] = args[1]
            elif Falcon:
                pass
            else:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        layer_name = f'model.layers.{i}'
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        print(f"get scales of layer {i}")
        act_scales = {}

        def stat_tensor(name, tensor):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()
            if name in act_scales:
                act_scales[layer_name + '.' + name] = torch.max(act_scales[name], comming_max)
            else:
                act_scales[layer_name + '.' + name] = comming_max

        def add_batch(name):
            def tmp(_, inp, out):
                stat_tensor(name, inp[0].data)
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if CHATGLM:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask, position_ids)[0]
                elif Falcon:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=None)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                if CHATGLM:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask, position_ids)[0]
                elif Falcon:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=None)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        if args.test_smooth:
            print(f"smoothing layer {i}")
            smooth_layer(layer_name, layer, act_scales, 0.8)

        if args.test_quant:
            print(f"quantizing layer {i}")
            quantize_layer(layer, nbits=args.nbits)

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    torch.cuda.empty_cache()

    return model
