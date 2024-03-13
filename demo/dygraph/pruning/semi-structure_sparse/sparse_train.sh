#!/bin/bash  

CUDA_VISIBLE_DEVICES=0,1 \
python3.7 -m paddle.distributed.launch \
--gpus="0,1" \
--log_dir="sparse_mobilenetv1_train_log" \
sparse_train.py \
    --model="MobileNetV1" \
    --data="imagenet" \
    --lr=0.1 \
    --batch_size=2 \
    --num_epochs=120 \
    --l2_decay=3e-5 \
    --lr_strategy="cosine_decay" \
    --model_path="./sparse_mobilenetv1_models"
