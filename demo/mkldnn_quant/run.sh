#!/bin/bash
MODEL_DIR=$1
DATA_FILE=$2
default_num_threads=1
default_with_accuracy=false
num_threads=${3:-$default_num_threads}
with_accuracy_layer=${4:-$default_with_accuracy}
ITERATIONS=500

GLOG_logtostderr=1 ./build/thread_sample \
    --infer_model=${MODEL_DIR} \
    --infer_data=${DATA_FILE} \
    --batch_size=1 \
    --num_threads=${num_threads} \
    --iterations=${ITERATIONS} \
    --with_accuracy_layer=${with_accuracy_layer} \
    --use_analysis=false
