# YOLOv6 TensorRT Benchmark测试（Linux）

## 环境准备

- CUDA、CUDNN：确认环境中已经安装CUDA和CUDNN，并且提前获取其安装路径。

- TensorRT：可通过NVIDIA官网下载[TensorRT 8.4.1.5](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.4.1/tars/tensorrt-8.4.1.5.linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz)或其他版本安装包。

- Paddle Inference C++预测库：编译develop版本请参考[编译文档](https://www.paddlepaddle.org.cn/inference/user_guides/source_compile.html)

## 编译可执行程序

- (1)修改`compile.sh`中依赖库路径，主要是以下内容：
```shell
# Paddle Inference预测库路径
LIB_DIR=/root/auto_compress/Paddle/build/paddle_inference_install_dir/
# CUDNN路径
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
# CUDA路径
CUDA_LIB=/usr/local/cuda/lib64
# TensorRT安装包路径
TENSORRT_ROOT=/root/auto_compress/trt/trt8.4/
```

## 测试

- FP32
```
./build/trt_run --model_file yolov6s_infer/model.pdmodel --params_file yolov6s_infer/model.pdiparams --run_mode=trt_fp32
```

- FP16
```
./build/trt_run --model_file yolov6s_infer/model.pdmodel --params_file yolov6s_infer/model.pdiparams --run_mode=trt_fp16
```

- INT8
```
./build/trt_run --model_file yolov6s_quant/model.pdmodel --params_file yolov6s_quant/model.pdiparams --run_mode=trt_int8
```

## 性能对比

| 模型  | 预测时延<sup><small>FP32</small><sup><br><sup>(ms) |预测时延<sup><small>FP16</small><sup><br><sup>(ms) | 预测时延<sup><small>INT8</small><sup><br><sup>(ms) |
| :-------- |:-------- |:--------: | :---------------------: |
| YOLOv6s |   9.06ms  |   2.90ms   |  1.83ms  |

环境：
- Tesla T4，TensorRT 8.4.1，CUDA 11.2
- batch_size=1
