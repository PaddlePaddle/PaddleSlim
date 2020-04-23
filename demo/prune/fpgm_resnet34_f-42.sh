#!/bin/bash  
export CUDA_VISIBLE_DEVICES=4,5,6,7
export FLAGS_fraction_of_gpu_memory_to_use=0.98
python slim_prune_train.py \
    --model="ResNet34" \
    --pretrained_model="/workspace/models/ResNet34_pretrained" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --lr_strategy="cosine_decay" \
    --criterion="geometry_median" \
    --model_path="./fpgm_resnet34_025_120_models" \
    2>&1 | tee fpgm_resnet025_120_train.log
    #--num_epochs=20 \
    #--test_period=5 \
    #--step_epochs 10 15 18 \
