import torch
import os
from loguru import logger

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
)
import argparse

from smoothquant.calibration import get_act_scales

def build_model_and_tokenizer(args, model_name):
    logger.info('load tokenizer')
    # while True:
    #     try:
    #         if "llama" in model_name:
    #             tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    #         else:
    #             tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    #         break
    #     except Exception as e:
    #         print(f"Error: {e}")
    # if "llama" in model_name:
    #     tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    logger.info('finish load tokenizer, load model')
    # while True:
    #     try:
    #         model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs, cache_dir=args.cache_dir)
    #         model.seqlen = args.seqlen
    #         break
    #     except Exception as e:
    #         print(f"Error: {e}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs, cache_dir=args.cache_dir)
    model.seqlen = args.seqlen
    logger.info('finish load model', flush=True)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='decapoda-research/llama-7b-hf', help='model name')
    parser.add_argument('--output-path', type=str, default='../act_scales/llama-7b-hf.pt',
                        help='where to save the act scales')
    parser.add_argument('--dataset-path', type=str, default='../dataset/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--seq-len', type=int, default=2048)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_name)

    # if not os.path.exists(args.dataset_path):
    #     print(f'Cannot find the dataset at {args.dataset_path}')
    #     print('Please download the Pile dataset and put the validation set at the path')
    #     print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
    #     raise FileNotFoundError

    act_scales = get_act_scales(model, tokenizer, args.dataset_path,
                                args.num_samples, args.seq_len)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)

def generate(model, tokenizer):
    args = parse_args()
    # model, tokenizer = build_model_and_tokenizer(args.model_name)

    # if not os.path.exists(args.dataset_path):
    #     print(f'Cannot find the dataset at {args.dataset_path}')
    #     print('Please download the Pile dataset and put the validation set at the path')
    #     print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
    #     raise FileNotFoundError

    act_scales = get_act_scales(model, tokenizer, args.dataset_path,
                                args.num_samples, args.seq_len)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)

if __name__ == '__main__':
    main()
