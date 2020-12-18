#!/bin/bash  
CUDA_VISIBLE_DEVICES=0,1 \
python -m paddle.distributed.launch \
--gpus="0,1" \
--log_dir="fpgm_mobilenetv2_train_log" \
train.py \
    --model="mobilenet_v2" \
    --data="imagenet" \
    --pruned_ratio=0.325 \
    --lr=0.001 \
    --num_epochs=90 \
    --test_period=5 \
    --step_epochs 30 60 80\
    --l2_decay=1e-4 \
    --lr_strategy="piecewise_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_mobilenetv2_models"
