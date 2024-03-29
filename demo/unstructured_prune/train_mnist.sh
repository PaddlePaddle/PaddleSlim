#!/bin/bash  
export CUDA_VISIBLE_DEVICES=2
export FLAGS_fraction_of_gpu_memory_to_use=0.98
python train.py \
          --batch_size=64 \
          --data="mnist" \
          --pruning_mode="threshold" \
          --threshold=0.01 \
          --lr=0.05
