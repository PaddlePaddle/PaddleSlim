#!/bin/bash  
python -m paddle.distributed.launch \
          --gpus='0,1,2,3' \
          --log_dir='log' \
          train.py \
          --batch_size 64 \
          --data imagenet \
          --pruning_mode ratio \
          --ratio 0.75 \
          --lr 0.005 \
          --num_epochs 108 \
          --step_epochs 71 88 \
          --initial_ratio 0.15 \
          --pruning_steps 100 \
          --stable_epochs 0 \
          --pruning_epochs 54 \
          --tunning_epochs 54 \
          --pruning_strategy gmp
