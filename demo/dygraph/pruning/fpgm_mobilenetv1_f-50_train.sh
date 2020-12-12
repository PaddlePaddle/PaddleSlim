#!/bin/bash  

CUDA_VISIBLE_DEVICES=0,1 \
python -m paddle.distributed.launch \
--gpus="0,1" \
--log_dir="fpgm_mobilenetv1_train_log" \
train.py \
    --model="mobilenet_v1" \
    --data="imagenet" \
    --pruned_ratio=0.3125 \
    --lr=0.1 \
    --num_epochs=120 \
    --test_period=10 \
    --step_epochs 30 60 90\
    --l2_decay=3e-5 \
    --lr_strategy="piecewise_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_mobilenetv1_models"
