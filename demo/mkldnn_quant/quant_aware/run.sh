#!/bin/bash
MODEL_DIR=/home/li/repo/Paddle/resnet50_quant_int8
DATA_FILE=$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin
num_threads=10
with_accuracy_layer=false
use_profile=true
ITERATIONS=100

./build/inference --logtostderr=1 \
    --infer_model=${MODEL_DIR} \
    --infer_data=${DATA_FILE} \
    --batch_size=1 \
    --num_threads=${num_threads} \
    --iterations=${ITERATIONS} \
    --with_accuracy_layer=${with_accuracy_layer} \
    --use_profile=${use_profile} \
