#!/bin/bash  
export CUDA_VISIBLE_DEVICES=3
python train.py \
          --batch_size=256 \
          --lr=0.05 \
          --threshold=0.01 \
          --pruning_mode="threshold" \
          --data="cifar10" \
