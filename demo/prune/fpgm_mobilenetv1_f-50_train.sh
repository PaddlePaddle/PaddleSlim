#!/bin/bash  
export CUDA_VISIBLE_DEVICES=0,1
export FLAGS_fraction_of_gpu_memory_to_use=0.98
python train.py \
    --model="MobileNet" \
    --pretrained_model="/workspace/models/MobileNetV1_pretrained" \
    --data="imagenet" \
    --pruned_ratio=0.3125 \
    --lr=0.1 \
    --num_epochs=120 \
    --test_period=10 \
    --step_epochs 30 60 90\
    --l2_decay=3e-5 \
    --lr_strategy="piecewise_decay" \
    --criterion="geometry_median" \
    --model_path="./fpgm_mobilenetv1_models" \
    2>&1 | tee fpgm_mobilenetv1_train.log
