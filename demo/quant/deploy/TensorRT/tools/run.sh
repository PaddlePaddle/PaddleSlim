#!/bin/bash
project_path=$(cd "$(dirname "$0")";pwd)
echo "${project_path}"
unset GREP_OPTIONS;
if [ ! -d "./build" ];then 
    echo -e "\033[33m run failed! \033[0m";
    echo -e "\033[33m you should build first \033[0m";
    exit;
fi

MODEL_DIR=MobileNetV1-quant # change to your model
BATCH_SIZE=1
USE_CALIB=false
USE_INT8=true
DATA_FILE=imagenet-eval-binary/0.data # change to your data file


build/trt_clas --model_dir=${MODEL_DIR} \
               --batch_size=${BATCH_SIZE} \
               --use_calib=${USE_CALIB} \
               --use_int8=${USE_INT8} \
               --data_file=${DATA_FILE} \
               --repeat_times=1

