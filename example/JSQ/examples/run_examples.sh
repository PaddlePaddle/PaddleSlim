# set environment
cuda_device=0
export CUDA_VISIBLE_DEVICES=$cuda_device

# nohup python examples.py \
#     --model huggyllama/llama-30b \
#     --prompt 'The extinction of the dinosaurs can be traced back a long time.' \
#     > prompts/llama-7b-dinner.log 2>&1 &

# nohup python examples.py \
#     --model /mnt/nvme0/wangzining/smooth2/examples/llama-7b-int7 \
#     --prompt 'The universe is the entirety of space, time, matter, and energy that exists.' \
#     > prompts/llama-7b-dinner.log 2>&1 &

# nohup python examples.py \
#     --model /mnt/nvme0/wangzining/smooth2/examples/llama-7b-int7 \
#     --prompt 'With the development of science and technology,' \
#     > prompts/llama-7b-dinner.log 2>&1 &

nohup python examples.py \
    --model meta-llama/Llama-2-13b-hf \
    --prompt 'In 2008, Beijing hosted the Olympic Games.' \
    > prompts/llama-7b-dinner.log 2>&1 &

# nohup python examples.py \
#     --model /mnt/nvme0/wangzining/smooth2/examples/llama2-7b \
#     --prompt 'The world is made of atoms.' \
#     > prompts/llama-7b-dinner.log 2>&1 &

# nohup python examples.py \
#     --model THUDM/chatglm3-6b-base \
#     --prompt 'Write a poem about life.' \
#     > prompts/llama-7b-dinner.log 2>&1 &

# nohup python examples.py \
#     --model /mnt/nvme0/wangzining/smooth2/examples/chatglm-6b \
#     --prompt 'Write a poem about life.' \
#     > prompts/llama-7b-dinner.log 2>&1 &