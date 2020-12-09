#!/bin/bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m paddle.distributed.launch \
--gpus="4,5,6,7" \
--log_dir="fpgm_resnet34_f-42_train_log" \
train.py \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --num_epochs=120 \
    --lr_strategy="cosine_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_resnet34_025_120_models"
#    &> fpgm_resnet025_120_train.log &
