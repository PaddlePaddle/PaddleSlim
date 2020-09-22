#!/bin/bash
MODEL_DIR=./mobilenetv2_INT8
DATA_FILE=/data/datasets/ImageNet_py/val.bin
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
    --use_analysis=false
