#!/bin/bash  
export CUDA_VISIBLE_DEVICES=3
python evaluate.py \
          --pruned_model="models" \
          --data="mnist"
