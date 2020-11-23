#!/bin/bash
PADDLE_LIB_PATH=trt_inference # change to your path
USE_GPU=ON
USE_MKL=ON
USE_TRT=ON
TENSORRT_INCLUDE_DIR=TensorRT-6.0.1.5/include # change to your path
TENSORRT_LIB_DIR=TensorRT-6.0.1.5/lib # change to your path

if [ $USE_GPU -eq ON ]; then
  export CUDA_LIB=`find /usr/local -name libcudart.so`
fi
rm -rf build
BUILD=build
mkdir -p $BUILD
cd $BUILD
cmake .. \
      -DPADDLE_LIB=${PADDLE_LIB_PATH} \
      -DWITH_GPU=${USE_GPU} \
      -DWITH_MKL=${USE_MKL} \
      -DCUDA_LIB=${CUDA_LIB} \
      -DUSE_TENSORRT=${USE_TRT} \
      -DTENSORRT_INCLUDE_DIR=${TENSORRT_INCLUDE_DIR} \
      -DTENSORRT_LIB_DIR=${TENSORRT_LIB_DIR}
make -j4
