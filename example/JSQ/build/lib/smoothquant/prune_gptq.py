import copy

import torch
import torch.nn as nn
from smoothquant.smooth import smooth_layer
from smoothquant.quantize import quantize_layer
from smoothquant.layerwrapper import WrappedGPT
from smoothquant.sparsegpt import SparseGPT
from smoothquant.gptq import GPTQ
from smoothquant.quant import Quantizer, Quant3Linear, make_quant3
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

# MMLU
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from categories import subcategories, categories
import time
choices = ["A", "B", "C", "D"]

def llama_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(subject, model, tokenizer, dev_df, test_df, ntrain=0):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")

        label = test_df.iloc[i, test_df.shape[1] - 1]
        # tmp = model(input_ids=input_ids).logits
        # logits = tmp[:, -1].flatten()
        breakpoint()
        logits = model(
            input_ids=input_ids,
        ).logits[:,-1].flatten()

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                        logits[tokenizer("D").input_ids[-1]],
                    ]
                ).float(),
                dim=0,
            )
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

def eval_mmlu(model, tokenizer, data_dir="/mnt/disk1/yg/benchmark_test/data", save_dir="results", ntrain=0, model_name="llama"):
    print(model)
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir,"mmlu", "results_{}".format(model_name))):
        os.makedirs(os.path.join(save_dir,"mmlu", "results_{}".format(model_name)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    all_time = 0
    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[: ntrain]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )
        eval_begin = time.time()
        cors, acc, probs = eval(subject, model, tokenizer, dev_df, test_df, ntrain=ntrain)
        eval_time = time.time() - eval_begin
        all_time += eval_time

        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(model_name)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(model_name, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                save_dir,"mmlu", "results_{}".format(model_name), "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))
    with open("./mmlures.txt", 'a') as f:
        f.write(f"Average accuracy: {weighted_acc}\n")

# End of MMLU


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
# from orisparsegpt.llama import llama_sparsegpt
def joint_pq(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if not args.test_sparsegpt and not args.test_wanda and not args.test_quant:
        eval_mmlu(model, tokenizer)
        return model
    print(args.sparsity_ratio)
    print(args.nbits)
    print(args.clip_h)
    # return llama_sparsegpt(model, tokenizer)
    if args.test_sparsegpt:
        return prune_sparsegpt(args, model, tokenizer, dev=torch.device("cuda:0"), prune_n=0, prune_m=0)
    else:
        return prune_wanda_and_gptq(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0)
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
            # print(name)
            # print(comming_max)
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
            smooth_layer(layer_name, layer, act_scales, 0.25) # Todo: changed!


        if args.test_quant:
            print(f"quantizing layer {i}")
            quantize_layer(layer, nbits=args.nbits)

        inps, outs = outs, inps

    torch.cuda.empty_cache()

    eval_mmlu(model, tokenizer)
    return model

from loguru import logger
def prepare_prune_wanda_and_gptq(args, device=torch.device("cuda:0")):
    logger.info('use gptq')
    model = build_model(args)
    layers = model.model.layers
    clip_opts = [0.00000,0.00004,0.00005,0.00006,0.00007]
    clip_table = [2] * len(layers)
    model = prune_wanda_and_gptq(model, args, clip_opts, clip_table, device)



def build_model(args):
    model_name = args.model
    if 'glm' in model_name:
        kwargs = {'torch_dtype': torch.float16, 'device_map': 'auto', 'trust_remote_code': True}
    else:
        kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.seqlen = args.seqlen
    return model

def prune_wanda_and_gptq(model, args, clip_opts, clip_table, device, prune_n = 0, prune_m = 0):
    # if not args.test_clip:
    #     args.clip_h = 0.0
    print(model)
    print("loading calibdation data")
    # dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    dataloader, testenc = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i]
        layer_name = f'model.layers.{i}'

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(
                dev), position_ids.to(dev)

        wrapped_layers = {}
        gptq = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.nbits, perchannel=True, sym=False, mse=False
            )
        print(f"pruning layer {i}")
        def add_batch(name):
            def tmp(_, inp, out):
                inp = clip_matrix(inp[0].data, args.abs, args.clip_l, clip_opts[clip_table[i]])
                wrapped_layers[name].add_batch(inp, out.data)
                gptq[name].add_batch(inp, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            weight = torch.abs(subset[name].weight.data)
            activation = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            # if args.test_ss:
            ss = generate_ss(wrapped_layers[name].inp_sum / wrapped_layers[name].inp_num, subset[name].weight.data)

            W_metric = weight * activation

            # if args.test_ss:
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

            # if args.test_wanda:
            subset[name].weight.data[W_mask] = 0  ## set weights to zero

            print(i, name)
            print('Quantizing ...')
            gptq[name].fasterquant(
                percdamp=.01, groupsize=-1, actorder=True,
                static_groups=False
            )
            quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
            gptq[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        del wrapped_layers
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    torch.cuda.empty_cache()
    # model = llama_pack3(model, quantizers)
    # torch.save(model.state_dict(), "GPTQ.pt")
    # eval_mmlu(model, tokenizer)
    model.save_pretrained(args.saved_path)
    return model


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    # if 'wikitext2' in name:
    #     return get_wikitext2(nsamples, seed, seqlen, model)
    # if 'ptb' in name:
    #     if 'new' in name:
    #         return get_ptb_new(nsamples, seed, seqlen, model)
    #     return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        # if 'new' in name:
        #     return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)

def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset, load_from_disk
    logger.info('load c4 datasets')
    # traindata = load_dataset(
    #     'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    # )
    # valdata = load_dataset(
    #     'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )
    logger.info('load from local')
    traindata = load_from_disk('/mnt/nvme0/wangzining/hf/allenai/c4/allenai--c4/train')
    valdata = load_from_disk('/mnt/nvme0/wangzining/hf/allenai/c4/allenai--c4/validation')
    # traindata = load_dataset(
    #     '/mnt/nvme0/wangzining/huggingface/datasets/c4/en/c4-train.00000-of-01024.json.gz', split='train'
    # )
    # valdata = load_dataset(
    #     '/mnt/nvme0/wangzining/huggingface/datasets/c4/en/c4-validation.00000-of-00008.json.gz', split='validation'
    # )
    from transformers import LlamaTokenizer, AutoTokenizer
    if 'glm' in model:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=False)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
    # tokenizer = LlamaTokenizer.from_pretrained('baffo32/decapoda-research-llama-7B-hf', use_fast=False)
    # tokenizer.save_pretrained("/mnt/nvme0/wangzining/smooth2/examples/llama-13b")
    # exit(0)
    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
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
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 


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

        # if args.test_smooth:
            # print(f"smoothing layer {i}")
        smooth_layer(layer_name, layer, act_scales, 0.8)

        # if args.test_quant:
        #     print(f"quantizing layer {i}")
        quantize_layer(layer, nbits=args.nbits)

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    torch.cuda.empty_cache()
    eval_mmlu(model, tokenizer)
    return model
