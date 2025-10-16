save_path="./log10.txt"
python_script="smoothquant_opt_demo.py"

if [ ! -e "$save_path" ]; then
  echo "文件不存在，正在创建..."
  touch "$save_path"
fi

cd ..
python setup.py build develop

cuda_device=0
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

# python smoothquant_opt_demo.py
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b' > llama-7b.log 2>&1 &
# nohup python smoothwanda.py yahma/llama-13b-hf c4 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-13b' > llama-13b.log 2>&1 &
# nohup python smoothwanda.py huggyllama/llama-30b c4 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-30b' > llama-30b.log 2>&1 &
# python smoothwanda.py huggyllama/llama-30b c4 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-30b'
# nohup python smoothwanda.py meta-llama/Llama-2-7b-hf c4 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama2-7b' > llama2-7b.log 2>&1 &
# nohup python smoothwanda.py meta-llama/Llama-2-13b-hf c4 --sparsity_ratio 0.4375 --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama13-7b' > llama2-13b.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --sparsity_ratio 0.55556 --rho 2.1 --nbits 6 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int6' > llama-7b-int6.log 2>&1 &

# python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --rho 2.1 
# python smoothwanda.py /mnt/nvme0/wangzining/smooth2/examples/save c4 --rho 2.1 
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --sparsity_ratio 0.5 --sparsity_type '2:4' --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-2-4' > llama-7b-2-4.log 2>&1 &
# nohup python smoothwanda.py baffo32/decapoda-research-llama-7B-hf c4 --sparsity_ratio 0.5 --sparsity_type '4:8' --rho 2.1 --nbits 8 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-4-8' > llama-7b-4-8.log 2>&1 &

# tokenizer
# python smoothwanda.py baffo32/decapoda-research-llama-7B-hf  c4 --rho 2.1 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b'
# python smoothwanda.py meta-llama/Llama-2-7b-chat-hf c4 --rho 2.1 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama2-7b'
# python smoothwanda.py yahma/llama-13b-hf  c4 --rho 2.1 --saved_path '/mnt/nvme0/wangzining/smooth2/examples/llama-13b'

# python wikitext2.py /mnt/nvme0/wangzining/smooth2/examples/llama2-7b-2-4 wikitext2
# python wikitext2.py /mnt/nvme0/wangzining/smooth2/examples/llama-13b wikitext2

python wikitext2.py /mnt/nvme0/wangzining/smooth2/examples/llama-7b-rho1 wikitext2
python wikitext2.py /mnt/nvme0/wangzining/smooth2/examples/llama-7b-rho2 wikitext2
python wikitext2.py /mnt/nvme0/wangzining/smooth2/examples/llama-7b-rho3 wikitext2
python wikitext2.py /mnt/nvme0/wangzining/smooth2/examples/llama-7b-rho4 wikitext2
python wikitext2.py /mnt/nvme0/wangzining/smooth2/examples/llama-7b-rho5 wikitext2