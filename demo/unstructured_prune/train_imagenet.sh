#!/bin/bash  
export CUDA_VISIBLE_DEVICES=2,3 
export FLAGS_fraction_of_gpu_memory_to_use=0.98
python3.7 train.py \
          --batch_size 512 \
          --data imagenet \
          --pruning_mode ratio \
          --ratio 0.55 \
          --lr 0.05 \
          --pretrained_model ./MobileNetV1_pretrained
