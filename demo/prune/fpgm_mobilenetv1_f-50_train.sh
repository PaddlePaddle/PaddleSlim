#!/bin/bash  
export CUDA_VISIBLE_DEVICES=4,5
export FLAGS_fraction_of_gpu_memory_to_use=0.98
python slim_prune_train.py \
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
    --model_path="./test_tice_fpgm_mobilenetv1_03_120_3e-5_models" \
    2>&1 | tee test_tice_fpgm_mobilenetv1_03_120_3e-5_train.log
    #--num_epochs=20 \
