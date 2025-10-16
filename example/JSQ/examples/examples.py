import argparse
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from loguru import logger


def evaluate(
    model,
    tokenizer,
    device=torch.device("cuda:0"),
    input=None,
    temperature=1,
    top_p=0.95,
    top_k=250,
    max_new_tokens=128,
    stream_output=False,
    **kwargs
):
    inputs = tokenizer(input, return_tensors='pt')
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_length=max_new_tokens,
                return_dict_in_generate=True,
            )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output


def main(args):
    # model = LlamaForCausalLM.from_pretrained(args.model)
    model_name = args.model
    if 'glm' in model_name:
        kwargs = {'torch_dtype': torch.float16, 'device_map': 'auto', 'trust_remote_code': True}
    else:
        kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    # model = LlamaForCausalLM.from_pretrained(model_name, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    # tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
    if 'glm' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False)
    # tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2
    model.eval()

    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     torch_dtype=torch.float16,
    #     device_map='auto'
    # )
    prompt = args.prompt
    sequences = evaluate(model, tokenizer, torch.device('cuda:0'), prompt)
    # sequences = pipeline(
    #     prompt,
    #     do_sample=True,
    #     top_k=10,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id,
	#     max_length=400,
    # )
    # for seq in sequences:
	#     logger.info(seq)
    logger.info(sequences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', 
                        type=str, 
                        help='LlaMa model to load; pass location of hugginface converted checkpoint.')
    parser.add_argument('--prompt', 
                        type=str, 
                        help='LlaMa model to load; pass location of hugginface converted checkpoint.')
    # parser.add_argument
    args = parser.parse_args()
    main(args)