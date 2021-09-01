#!/bin/bash  
python -m paddle.distributed.launch \
          --gpus='0,1,2,3' \
          --log_dir='log' \
          train.py \
          --batch_size 64 \
          --data imagenet \
          --pruning_mode ratio \
          --ratio 0.55 \
          --lr 0.05
