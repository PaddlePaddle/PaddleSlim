#!/bin/bash  
export CUDA_VISIBLE_DEVICES=2,3 
export FLAGS_fraction_of_gpu_memory_to_use=0.98
python3.7 train.py \
          --batch_size=256 \
          --data="mnist" \
          --pruning_mode="threshold" \
          --ratio=0.45 \
          --threshold=1e-5 \
          --lr=0.075 \
