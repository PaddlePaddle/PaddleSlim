#!/bin/bash
#MODEL_DIR=$HOME/repo/Paddle/resnet50_quant_int8
MODEL_DIR=/home/li/repo/PaddleSlim/demo/mkldnn_quant/quant_aware/20200608_mobilenetv2_INT8_MODEL
DATA_FILE=$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin
num_threads=1
with_accuracy_layer=false
use_profile=true
ITERATIONS=0

./build/inference --logtostderr=1 \
    --infer_model=${MODEL_DIR} \
    --infer_data=${DATA_FILE} \
    --batch_size=1 \
    --num_threads=${num_threads} \
    --iterations=${ITERATIONS} \
    --with_accuracy_layer=${with_accuracy_layer} \
    --use_profile=${use_profile} \
    --optimize_fp32_model=false
