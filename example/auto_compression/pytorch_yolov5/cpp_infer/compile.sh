#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

mkdir -p build
cd build
rm -rf *

DEMO_NAME=trt_run

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=ON

LIB_DIR=/root/auto_compress/Paddle/build/paddle_inference_install_dir/
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/root/auto_compress/trt/trt8.4/

WITH_ROCM=OFF
ROCM_LIB=/opt/rocm/lib

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DWITH_ROCM=${WITH_ROCM} \
  -DROCM_LIB=${ROCM_LIB} \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB} \
  -DTENSORRT_ROOT=${TENSORRT_ROOT}

make -j
