#!/bin/bash
unset GREP_OPTIONS;
if [ ! -d "./build" ];then 
    echo -e "\033[33m run failed! \033[0m";
    echo -e "\033[33m you should build first \033[0m";
    exit;
fi

MODEL_DIR=MobileNetV1-quant # change to your model_dir
DATA_DIR=imagenet-eval-binary # chage to your data_dir
USE_INT8=True

./build/test_acc --model_dir=$MODEL_DIR --data_dir=$DATA_DIR --int8=$USE_INT8

