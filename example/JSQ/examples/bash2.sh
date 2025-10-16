save_path="./log10.txt"
python_script="smoothquant_opt_demo.py"

if [ ! -e "$save_path" ]; then
  echo "文件不存在，正在创建..."
  touch "$save_path"
fi

cd ..
python setup.py build develop

cuda_device=4
# set environment
export CUDA_VISIBLE_DEVICES=$cuda_device

cd examples/

#clip_hs=(0.0001 0.001 0.005 0.01 0.02 0.03 0.0 0.05 0.1)
clip_hs=(0.05)
#rhos=(0 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1)
# for rho in "${rhos[@]}"; do
#     echo "Params: $rho"
#     python smoothquant_opt_demo.py --rho "$rho" >> "$save_path"
# done
# for clip_h in "${clip_hs[@]}"; do
#    echo "Params: $clip_h"
#    python smoothquant_opt_demo.py --clip_h "$clip_h" --rho 0.001 
# done
nohup python smoothwanda.py /root/.paddlenlp/models/facebook/llama-7b wikitext2 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/disk3/cym/JSQ-paddle/llama-7b-paddle-1' > llama-7b-paddle-v9.log 2>&1 &
# python smoothquant_opt_demo.py
# nohup python smoothwanda.py /mnt/disk1/hg/huggingface/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348 wikitext2 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/disk3/cym/JSQ/llama-7b-new' > llama-7b.log 2>&1 &
# nohup python smoothwanda.py /mnt/disk1/hg/huggingface/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348 wikitext2 --sparsity_ratio 0.5 --sparsity_type '2:4' --rho 2.1 --nbits 8 --saved_path '/mnt/disk3/cym/JSQ/llama-7b-24' > llama-7b-24.log 2>&1 &
# nohup python smoothwanda.py /mnt/disk1/hg/huggingface/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348 wikitext2 --sparsity_ratio 0.5 --sparsity_type '4:8' --rho 2.1 --nbits 8 --saved_path '/mnt/disk3/cym/JSQ/llama-7b-48' > llama-7b-48.log 2>&1 &

# nohup python smoothwanda.py /mnt/disk1/yg/llama/llama-2-7b-hf wikitext2 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/disk3/cym/JSQ/llama-2-7b-new' > llama2-7b.log 2>&1 &
# nohup python smoothwanda.py /mnt/disk1/yg/llama/llama-2-7b-hf wikitext2 --sparsity_ratio 0.5 --sparsity_type '2:4' --rho 2.1 --nbits 8 --saved_path '/mnt/disk3/cym/JSQ/llama2-7b-24' > llama2-7b-24.log 2>&1 &
# nohup python smoothwanda.py /mnt/disk1/yg/llama/llama-2-7b-hf wikitext2 --sparsity_ratio 0.5 --sparsity_type '4:8' --rho 2.1 --nbits 8 --saved_path '/mnt/disk3/cym/JSQ/llama2-7b-48' > llama2-7b-48.log 2>&1 &

# nohup python smoothwanda.py /mnt/disk1/wjy/Llama-2-13b-hf wikitext2 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/disk3/cym/JSQ/llama-2-13b-new' > llama2-13b.log 2>&1 &
# nohup python smoothwanda.py /mnt/disk1/hg/huggingface/cache/models--decapoda-research--llama-13b-hf/snapshots/438770a656712a5072229b62256521845d4de5ce wikitext2 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/disk3/cym/JSQ/llama-13b-new' > llama-13b.log 2>&1 &
# > llama-7b.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --test_only --sparsity_ratio 0.4375 --rho 10 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-rho10' > llama-7b-rho10.log 2>&1 &
# nohup python smoothwanda.py yahma/llama-13b-hf c4 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-13b' > llama-13b.log 2>&1 &
# nohup python smoothwanda.py huggyllama/llama-30b c4 --test_only --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-30b' > llama-30b.log 2>&1 &
# python smoothwanda.py huggyllama/llama-30b c4 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-30b'
# nohup python smoothwanda.py meta-llama/Llama-2-7b-hf c4 --sparsity_ratio 0.5 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama2-7b-0-5' > llama2-7b-0-5.log 2>&1 &
# nohup python smoothwanda.py meta-llama/Llama-2-13b-hf c4 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama13-7b' > llama2-13b.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --sparsity_ratio 0.55556 --rho 2.1 --nbits 6 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int6' > llama-7b-int6.log 2>&1 &
# nohup python smoothwanda.py THUDM/chatglm3-6b-base c4 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/chatglm-6b' > chatglm-6b.log 2>&1 &
# nohup python smoothwanda.py --test_only THUDM/chatglm3-6b-base c4 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/chatglm-6b' > chatglm-6b.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-rho20' > llama-7b-rho20.log 2>&1 &
# nohup python smoothwanda.py /mnt/nvme0/llm_weights/llama2/llama2-70b c4 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining2/JSQ/examples/llama2-70b' > llama2-70b.log 2>&1 &
# python smoothwanda.py /mnt/disk1/yg/llama/llama-2-7b-hf/ wikitext2 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 
# nohup python smoothwanda.py /mnt/nvme0/wangzining2/CodeLlama/CodeLlama-7b-hf c4 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining2/JSQ/examples/llama-7b-code' > llama-7b-code.log 2>&1 &


# python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --rho 2.1 
# python smoothwanda.py /mnt/nvme0/wangzining/smooth2/examples/save c4 --rho 2.1 
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --sparsity_ratio 0.5 --sparsity_type '2:4' --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-2-4' > llama-7b-2-4.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --sparsity_ratio 0.5 --sparsity_type '4:8' --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-4-8' > llama-7b-4-8.log 2>&1 &
# nohup python smoothwanda.py meta-llama/Llama-2-7b-hf c4 --sparsity_ratio 0.5 --sparsity_type '2:4' --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama2-7b-2-4' > llama2-7b-2-4.log 2>&1 &
# nohup python smoothwanda.py meta-llama/Llama-2-7b-hf c4 --sparsity_ratio 0.5 --sparsity_type '4:8' --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama2-7b-4-8' > llama2-7b-4-8.log 2>&1 &



# tokenizer
# python smoothwanda.py baffo32/decapoda-research-llama-7B-hf  c4 --rho 2.1 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b'
# python smoothwanda.py meta-llama/Llama-2-7b-chat-hf c4 --rho 2.1 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama2-7b'
# python smoothwanda.py yahma/llama-13b-hf  c4 --rho 2.1 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-13b'
# python smoothwanda.py meta-llama/Llama-2-7b-hf c4 --rho 2.1 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama2-7b' 
# python smoothwanda.py meta-llama/Llama-2-13b-hf c4 --rho 2.1 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama13-7b' 


# int7 test
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --test_only --sparsity_ratio 0.265625 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int7' > llama-7b-int7.log 2>&1 &

# int5 test
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --test_only --sparsity_ratio 0.609375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int5-1' > llama-7b-int5-1.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --test_only --sparsity_ratio 0.305556 --rho 2.1 --nbits 6 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int5-2' > llama-7b-int5-2.log 2>&1 &

# int3 test
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --test_only --sparsity_ratio 0.75 --rho 2.1 --nbits 6 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int3-1' > llama-7b-int3-1.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --test_only --sparsity_ratio 0.64 --rho 2.1 --nbits 5 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int3-2' > llama-7b-int3-2.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --test_only --sparsity_ratio 0.4375 --rho 2.1 --nbits 4 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int3-3' > llama-7b-int3-3.log 2>&1 &

# int2 test
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --test_only --sparsity_ratio 0.88889 --rho 2.1 --nbits 6 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int2-1' > llama-7b-int2-1.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --test_only --sparsity_ratio 0.84 --rho 2.1 --nbits 5 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int2-2' > llama-7b-int2-2.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --test_only --sparsity_ratio 0.75 --rho 2.1 --nbits 4 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int2-3' > llama-7b-int2-3.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --test_only --sparsity_ratio 0.55556 --rho 2.1 --nbits 3 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int2-4' > llama-7b-int2-4.log 2>&1 &
# nohup python smoothwanda.py huggyllama/llama-30b c4 --test_only --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-30b' > llama-30b.log 2>&1 &
# python smoothwanda.py huggyllama/llama-30b c4 --test_only --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-30b'

# activation distribution
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --test_only --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-activation' > llama-7b-activation.log 2>&1 &
# python smoothwanda.py '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-rho10' c4 --test_only --sparsity_ratio 0.4375 --rho 10 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-activation-rho10'

# gptq
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --gptq --test_only --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-gptq' > llama-7b-gptq.log 2>&1 &

# ablation
# ours
# nohup python smoothwanda.py /mnt/disk1/hg/huggingface/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348 wikitext2 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/disk3/cym/JSQ/llama-7b-ablation' > llama-7b-ablation.log 2>&1 &
# w/o SAR
# nohup python smoothwanda.py /mnt/disk1/hg/huggingface/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348 wikitext2 --sparsity_ratio 0.4375 --rho 0 --nbits 8 --saved_path '/mnt/disk3/cym/JSQ/llama-7b' > llama-7b-norho.log 2>&1 &
# w/o clip
# nohup python smoothwanda.py /mnt/disk1/hg/huggingface/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348 wikitext2 --test_only --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/disk3/cym/JSQ/llama-7b-noclip' > llama-7b-noclip.log 2>&1 &
# w/o anneal
# nohup python smoothwanda.py /mnt/disk1/hg/huggingface/cache/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348 wikitext2 --test_only --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/disk3/cym/JSQ/llama-7b-noanneal' > llama-7b-noanneal.log 2>&1 &


# rho
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --sparsity_ratio 0.4375 --rho 1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-rho1' > llama-7b-rho1.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --sparsity_ratio 0.4375 --rho 2 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-rho2' > llama-7b-rho2.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --sparsity_ratio 0.4375 --rho 3 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-rho3' > llama-7b-rho3.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --sparsity_ratio 0.4375 --rho 4 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-rho4' > llama-7b-rho4.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --sparsity_ratio 0.4375 --rho 5 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-rho5' > llama-7b-rho5.log 2>&1 &

# calibration
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --nsamples 32 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-cali32' > llama-7b-cali32.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --nsamples 64 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-cali64' > llama-7b-cali64.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --nsamples 16 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-cali16' > llama-7b-cali16.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --nsamples 8 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-cali8' > llama-7b-cali8.log 2>&1 &