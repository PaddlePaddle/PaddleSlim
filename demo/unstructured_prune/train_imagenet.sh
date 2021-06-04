#!/bin/bash  
# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# export FLAGS_fraction_of_gpu_memory_to_use=0.98
python3.7 train_parameters.py \
          --batch_size 256 \
          --data imagenet \
          --model MobileNetSensitive30 \
          --pruning_mode ratio \
          --ratio 0.70 \
          --initial_ratio 0.10 \
          --lr 0.005 \
          --pretrained_model MobileNetV1_sensitive-30 \
          --num_epochs 120 \
          --test_period 5 \
          --model_path "./models" \
          --stable_epochs 0 \
          --pruning_epochs 28 \
          --tunning_epochs 16
