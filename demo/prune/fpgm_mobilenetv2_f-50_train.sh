#!/bin/bash  
export CUDA_VISIBLE_DEVICES=0,1
export FLAGS_fraction_of_gpu_memory_to_use=0.98
python train.py \
    --model="MobileNetV2" \
    --pretrained_model="/workspace/models/MobileNetV2_pretrained" \
    --data="imagenet" \
    --pruned_ratio=0.325 \
    --lr=0.001 \
    --num_epochs=90 \
    --test_period=5 \
    --step_epochs 30 60 80\
    --l2_decay=1e-4 \
    --lr_strategy="piecewise_decay" \
    --criterion="geometry_median" \
    --model_path="./fpgm_mobilenetv2_models" \
    2>&1 | tee fpgm_mobilenetv2_train.log
