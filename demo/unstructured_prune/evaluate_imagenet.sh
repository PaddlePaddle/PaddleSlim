#!/bin/bash  
export CUDA_VISIBLE_DEVICES=3
export FLAGS_fraction_of_gpu_memory_to_use=0.98
python evaluate.py \
          --pruned_model="models" \
          --data="imagenet"
