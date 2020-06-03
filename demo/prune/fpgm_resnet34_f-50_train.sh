#!/bin/bash  
export CUDA_VISIBLE_DEVICES=0,1
export FLAGS_fraction_of_gpu_memory_to_use=0.98
python train.py \
    --model="ResNet34" \
    --pretrained_model="/workspace/models/ResNet34_pretrained" \
    --data="imagenet" \
    --pruned_ratio=0.3125 \
    --lr=0.001 \
    --num_epochs=70 \
    --test_period=5 \
    --step_epochs 30 60 \
    --l2_decay=1e-4 \
    --lr_strategy="piecewise_decay" \
    --criterion="geometry_median" \
    --model_path="./fpgm_resnet34_models" \
    2>&1 | tee fpgm_resnet03_train.log
