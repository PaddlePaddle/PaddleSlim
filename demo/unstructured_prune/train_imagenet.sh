#!/bin/bash  
# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# export FLAGS_fraction_of_gpu_memory_to_use=0.98
python3.7 train_parameters.py \
          --batch_size 1024 \
          --data imagenet \
          --pruning_mode ratio \
          --ratio 0.85 \
          --initial_ratio 0.05 \
          --lr 0.005 \
          --pretrained_model MobileNetV1_distilled \
          --num_epochs 120 \
          --test_period 5 \
          --model_path "./models-GMP"
