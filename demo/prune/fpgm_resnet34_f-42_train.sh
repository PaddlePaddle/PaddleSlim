#!/bin/bash  
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fraction_of_gpu_memory_to_use=0.98
python train.py \
    --model="ResNet34" \
    --pretrained_model="/workspace/models/ResNet34_pretrained" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --lr_strategy="cosine_decay" \
    --criterion="geometry_median" \
    --model_path="./fpgm_resnet34_025_120_models" \
    2>&1 | tee fpgm_resnet025_120_train.log
