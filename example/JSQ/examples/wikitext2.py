import argparse
from smoothquant.test import prune_wanda_and_smoothquant_annealing
from loguru import logger
import torch
import torch.nn as nn
from datasets import load_dataset
import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
)


parser = argparse.ArgumentParser()
parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
parser.add_argument(
    'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
    help='Where to extract calibration data from.'
)
# parser.add_argument('--model', default="decapoda-research/llama-7b-hf", type=str, help='LLaMA model')
parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
parser.add_argument('--seqlen', type=int, default=2048)
parser.add_argument('--sparsity_ratio', type=float, default=0.4375, help='Sparsity level')
parser.add_argument("--sparsity_type", default="unstructured", type=str, choices=["unstructured", "4:8", "2:4"])
parser.add_argument("--prune_method", default="wanda", type=str, choices=["magnitude", "wanda", "sparsegpt"])
parser.add_argument("--cache_dir", default="/mnt/disk1/hg/huggingface/cache", type=str)
parser.add_argument('--use_variant', action="store_true",
                    help="whether to use the wanda variant described in the appendix")
parser.add_argument('--save', type=str, default=None, help='Path to save results.')
parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
parser.add_argument('--clip_l', type=float, default=0.0)
parser.add_argument('--clip_h', type=float, default=0.001)
parser.add_argument('--abs', action="store_false")
parser.add_argument('--saved_path', type=str)
parser.add_argument('--rho', type=float, default=0.0)
parser.add_argument('--nbits', type=int, default=8)
args = parser.parse_args()

# annealing
# prune_wanda_and_smoothquant_annealing(args)


def build_model(args):
    model_name = args.model
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    while True:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            model.seqlen = args.seqlen
            break
        except Exception as e:
            logger.info(f"Error: {e}")
    return model

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
                model.seqlen = args.seqlen
                break
            except Exception as e:
                print(f"Error: {e}")
        return model, tokenizer

def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    
    # traindata = load_dataset('EleutherAI/wikitext_document_level', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('EleutherAI/wikitext_document_level', 'wikitext-2-raw-v1', split='test')

    # trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['page']), return_tensors='pt')

    random.seed(seed)
    
    return None, testenc

@torch.no_grad()
def llama_eval(model, testenc, dev=torch.device("cuda:0")):
    
    # logger.info('Evaluating ...')
    model.eval()
    model.seqlen = 280
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        # print(i)
        layer = layers[i].to(dev)
        
        # if args.nearest:
        #     subset = find_layers(layer)
        #     for name in subset:
        #         quantizer = Quantizer()
        #         quantizer.configure(
        #             args.wbits, perchannel=True, sym=False, mse=False
        #         )
        #         W = subset[name].weight.data
        #         quantizer.find_params(W, weight=True)
        #         subset[name].weight.data = quantize(
        #             W, quantizer.scale, quantizer.zero, quantizer.maxq
        #         ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        # logger.info(f'{i}th sample')
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    # logger.info(ppl.item())

    model.config.use_cache = use_cache
    return ppl


model = build_model(args)

dataloader, testenc = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
logger.info("dataset loading complete")
ppl = llama_eval(model, testenc)
logger.info(f'ppl is {ppl}')