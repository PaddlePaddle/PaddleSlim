# YOLOv7 TensorRT Benchmark测试（Linux）

## 环境准备

- CUDA、CUDNN：确认环境中已经安装CUDA和CUDNN，并且提前获取其安装路径。

- TensorRT：可通过NVIDIA官网下载[TensorRT 8.6.1.6](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz)或其他版本安装包。

- Paddle Inference C++预测库：编译develop版本请参考[编译文档](https://www.paddlepaddle.org.cn/inference/user_guides/source_compile.html)。编译完成后，会在build目录下生成`paddle_inference_install_dir`文件夹，这个就是我们需要的C++预测库文件。
或在官网下载[c++推理库](https://www.paddlepaddle.org.cn/inference/v2.6/guides/install/download_lib.html)

## 编译可执行程序

- (1)修改`compile.sh`中依赖库路径，主要是以下内容：
```shell
# Paddle Inference预测库路径
LIB_DIR=/work/Paddle/build/paddle_inference_install_dir
# CUDNN路径
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
# CUDA路径
CUDA_LIB=/usr/local/cuda/lib64
# TensorRT安装包路径，为TRT资源包解压完成后的绝对路径，其中包含`lib`和`include`文件夹
TENSORRT_ROOT=/work/TensorRT-8.6.1.6
```

## Paddle tensorRT测试

- YOLOv5
```
# FP32
./build/trt_run --model_file yolov5_model/inference_model/model.pdmodel --params_file yolov5_model/inference_model/model.pdiparams --run_mode=trt_fp32
# FP16
./build/trt_run --model_file yolov5_model/inference_model/model.pdmodel --params_file yolov5_model/inference_model/model.pdiparams --run_mode=trt_fp16
# INT8
./build/trt_run --model_file yolov5s_quantaware/model.pdmodel --params_file yolov5s_quantaware/model.pdiparams --run_mode=trt_int8
```

- YOLOv6
```
# FP32
./build/trt_run --model_file yolov6_model/inference_model/model.pdmodel --params_file yolov6_model/inference_model/model.pdiparams --run_mode=trt_fp32
# FP16
./build/trt_run --model_file yolov6_model/inference_model/model.pdmodel --params_file yolov6_model/inference_model/model.pdiparams --run_mode=trt_fp16
# INT8
./build/trt_run --model_file yolov6s_quantaware/model.pdmodel --params_file yolov6s_quantaware/model.pdiparams --run_mode=trt_int8
```


- YOLOv7
```
# FP32
./build/trt_run --model_file yolov7-tiny/inference_model/model.pdmodel --params_file yolov7-tiny/inference_model/model.pdiparams --run_mode=trt_fp32
# FP16
./build/trt_run --model_file yolov7-tiny/inference_model/model.pdmodel --params_file yolov7-tiny/inference_model/model.pdiparams --run_mode=trt_fp16
# INT8
./build/trt_run --model_file yolov7-quantAware/model.pdmodel --params_file yolov7-quantAware/model.pdiparams --run_mode=trt_int8
```

## 原生TensorRT测试

```shell
# FP32
trtexec --onnx=yolov5s.onnx --workspace=1024 --avgRuns=1000 --inputIOFormats=fp32:chw --outputIOFormats=fp32:chw
# FP16
trtexec --onnx=yolov5s.onnx --workspace=1024 --avgRuns=1000 --inputIOFormats=fp32:chw --outputIOFormats=fp32:chw --fp16
# INT8
trtexec --onnx=yolov5s.onnx --workspace=1024 --avgRuns=1000 --inputIOFormats=fp32:chw --outputIOFormats=fp32:chw --int8
```
- 注：可把--onnx=yolov5s.onnx替换成yolov6s.onnx和yolov7.onnx模型
