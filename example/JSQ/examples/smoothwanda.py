import argparse
from jsq import prune_wanda_and_smoothquant_annealing
# from smoothquant.prune_gptq import prepare_prune_wanda_and_gptq
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
parser.add_argument('--nsamples', type=int, default=1, help='Number of calibration samples.')
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
parser.add_argument('--saved_path', type=str, default="./outputs/model_pruned")
parser.add_argument('--rho', type=float, default=0.0)
parser.add_argument('--nbits', type=int, default=8)
parser.add_argument('--test_only', action="store_true", default=False )
parser.add_argument('--gptq', action="store_true", default=False)

args = parser.parse_args()

# annealing
if args.gptq:
    # prepare_prune_wanda_and_gptq(args)
    pass
else:
    prune_wanda_and_smoothquant_annealing(args)



