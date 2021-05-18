#!/bin/bash  
export CUDA_VISIBLE_DEVICES=2,3 
export FLAGS_fraction_of_gpu_memory_to_use=0.98
python3.7 train.py \
          --batch_size=512 \
          --data="mnist" \
          --pruning_mode="threshold" \
          --threshold=0.01 \
          --lr=0.05 \
