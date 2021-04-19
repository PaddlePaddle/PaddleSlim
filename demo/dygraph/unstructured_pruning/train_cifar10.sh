#!/bin/bash  
export CUDA_VISIBLE_DEVICES=3
python3.7 train.py \
          --batch_size=128 \
          --lr=0.05 \
          --ratio=0.45 \
          --threshold=1e-5 \
          --pruning_mode="threshold" \
          --data="cifar10" \
