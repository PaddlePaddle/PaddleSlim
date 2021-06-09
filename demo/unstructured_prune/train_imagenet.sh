#!/bin/bash  
# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# export FLAGS_fraction_of_gpu_memory_to_use=0.98
python3.7 train_parameters.py \
          --batch_size 256 \
          --data imagenet \
          --pruning_mode ratio \
          --ratio 0.8 \
          --lr 0.0005 \
          --pretrained_model modelsS30-singlegpu \
          --model MobileNetSensitive30 \
          --num_epochs 30 \
          --test_period 5 \
          --model_path "./modelsS30-singlegpu" \
          --initial_ratio 0.80 \
          --stable_epochs 0 \
          --pruning_epochs 0 \
          --tunning_epochs 30 \
          --ratio_steps_per_epoch 2 \
          --step_epochs 15
