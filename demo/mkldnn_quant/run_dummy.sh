#!/bin/bash
MODEL_DIR=$1
DATA_FILE=$2
default_num_threads=1
num_threads=${3:-$default_num_threads}
default_batch_size=1
batch_size=${4:-default_batch_size}
default_with_accuracy=false
with_accuracy_layer=${5:-$default_with_accuracy}
default_with_analysis=true
with_analysis=${6:-$default_with_analysis}
default_enable_mkldnn_bfloat16=false
with_mkldnn_bfloat16=${7:-$default_enable_mkldnn_bfloat16}
ITERATIONS=0

GLOG_logtostderr=1 ./build/sample_tester_fake_data \
    --infer_model=${MODEL_DIR} \
    --infer_data=${DATA_FILE} \
    --batch_size=${batch_size} \
    --num_threads=${num_threads} \
    --iterations=${ITERATIONS} \
    --with_accuracy_layer=${with_accuracy_layer} \
    --use_analysis=${with_analysis} \
    --enable_mkldnn_bfloat16=${with_mkldnn_bfloat16}

# KMP_BLOCKTIME=1 KMP_SETTINGS=1 KMP_AFFINITY=granularity=fine,verbose,compact,1,0 numactl bash run_dummy.sh INT8 val_100.bin 4 10 false false false
