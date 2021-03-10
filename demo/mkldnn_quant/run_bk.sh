#!/bin/bash
mkdir build && cd build
PADDLE_LIB=${PADDLE_LIB}

cmake .. -DPADDLE_LIB=${PADDLE_LIB} && make -j

MODEL_DIR=/home/li/models/ResNet50_4th_qat_int8
DATA_FILE=/mnt/disk500/data/int8_full_val.bin
num_threads=1
with_accuracy_layer=false
use_profile=true
ITERATIONS=0

GLOG_logtostderr=1 ./build/sample_tester \
    --infer_model=${MODEL_DIR} \
    --infer_data=${DATA_FILE} \
    --batch_size=1 \
    --num_threads=${num_threads} \
    --iterations=${ITERATIONS} \
    --with_accuracy_layer=${with_accuracy_layer} \
    --use_profile=${use_profile} \
    --use_analysis=false
