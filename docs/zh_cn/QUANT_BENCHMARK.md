# 模型量化模型库

## 图像分类

| 模型 | 示例 | Top-1 Acc<sup>FP32</sup> | Top-1 Acc<sup>INT8</sup>  | 模型体积<sup>FP32</sup> | 模型体积<sup>INT8</sup> | TRT<sup>FP32</sup> | TRT<sup>FP16</sup> | TRT<sup>INT8</sup> | CPU<sup>FP32</sup> | CPU<sup>INT8</sup> | ARM-CPU<sup>FP32</sup>| ARM-CPU<sup>INT8</sup> |  
|:---:|:---:|:----:|:---:|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|:---:|:-----:|:-----:|  
| ResNet50-vd | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/image_classification) | 79.08 | 78.17 | 98MB | 25MB | 7.0ms | 2.4ms | 1.6ms | 77.7ms | 24.7ms | - | - |
| MobileNetV3_large_x1_0 | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/image_classification) | 75.32 | 74.04 | 22MB | 5.6MB | 2.3ms | 1.4ms | 1.1ms | 11.3ms | 7.7ms | 16.6ms | 9.8ms |
| PPLCNetV2 | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/image_classification) | 76.63 | 75.87 | 26MB | 6.7MB | 2.1ms | 1.2ms | 0.8ms | 16.3ms | 7.7ms | 36.5ms | 15.8ms |
| PPHGNet_tiny | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/image_classification) | 79.59 | 78.95 | 57MB | 15MB | 6.4ms | 2.6ms | 1.7ms | 75.3ms | 32.8ms | - | - |
| EfficientNetB0 | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/image_classification) | 77.02 | 74.64 | 21MB | 5.7MB | 3.6ms | 2.1ms | 1.7ms | 29.5ms | - | - | - |

测试环境：
- GPU: Tesla T4; cuda11.1/cudnn8.1.1/trt8.4.0.6;
- CPU: Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz
- ARM CPU 测试环境：SDM865(4xA77+4xA55)
- PaddlePaddle 2.4.0
- CPU测试开启MKLDNN，单线程
- 以上数据使用[PaddleTest](https://github.com/PaddlePaddle/PaddleTest/tree/develop/inference/python_api_test/test_int8_model)的Python脚本测试得到


## 目标检测

| 模型 | 示例 | mAP(0.5:0.95)<sup>FP32</sup> | mAP(0.5:0.95)<sup>INT8</sup>  | 模型体积<sup>FP32</sup> | 模型体积<sup>INT8</sup> | TRT<sup>FP32</sup> | TRT<sup>FP16</sup> | TRT<sup>INT8</sup> | CPU<sup>FP32</sup> | CPU<sup>INT8</sup> | ARM-CPU<sup>FP32</sup>| ARM-CPU<sup>INT8</sup> |  
|:---:|:---:|:----:|:---:|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|:---:|:-----:|:-----:|  
| PP-YOLOE-l | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/detection) | 50.9 | 50.6 | 98MB | 25MB | - | - | - | 1526ms | 1081ms | - | - |
| PP-PicoDet | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/detection) | 50.9 | 50.6 | 98MB | 25MB | - | 6.8ms | 6.1ms | 56.9ms | 39.0ms | - | - |
| YOLOv5s | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/pytorch_yolo_series) | 37.4 | 36.9 | 28.1MB | 7.4MB | 14.5ms | 8.5ms | 7.4ms | 310.5ms | 265.7ms | - | - |
| YOLOv6s | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/pytorch_yolo_series) | 42.4 | 41.3 | 65.9MB | 16.8MB | 17.4ms | 7.4ms | 5.3ms | 387.3ms | 157.2ms | - | - |
| YOLOv7 | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/pytorch_yolo_series) | 51.1 | 50.8 | 141.4MB | 35.8MB | 48.2 | 17.0ms | 12.4ms | 1268.6ms | 864.3ms | - | - |

测试环境：
- GPU: Tesla T4; cuda11.1/cudnn8.1.1/trt8.4.0.6;
- CPU: Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz
- ARM CPU 测试环境：SDM865(4xA77+4xA55)
- PaddlePaddle 2.4.0
- CPU测试开启MKLDNN，单线程
- 以上数据使用[PaddleTest](https://github.com/PaddlePaddle/PaddleTest/tree/develop/inference/python_api_test/test_int8_model)的Python脚本测试得到，在COCO val2017验证集上测试

## 语义分割

| 模型 | 示例 | mIoU<sup>FP32</sup> | mIoU<sup>INT8</sup>  | 模型体积<sup>FP32</sup> | 模型体积<sup>INT8</sup> | TRT<sup>FP32</sup> | TRT<sup>FP16</sup> | TRT<sup>INT8</sup> | CPU<sup>FP32</sup> | CPU<sup>INT8</sup> | ARM-CPU<sup>FP32</sup>| ARM-CPU<sup>INT8</sup> |  
|:---:|:---:|:----:|:---:|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|:---:|:-----:|:-----:|  
| PP-HumanSeg-Lite | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/semantic_segmentation) | 96.0 | 95.94 | 0.54MB | 0.2MB | 2.7ms | 2.1ms | 1.9ms | 63.8ms | 59.6ms | 56.36ms | 49.65ms |
| PP-Liteseg | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/semantic_segmentation) | 77.04 | 76.99 | 31MB | 7.8MB | 46.4ms | 30.4ms | 27.1ms | 1289ms | 785ms | - | - |
| HRNet | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/semantic_segmentation) | 78.97 | 78.46 | 37MB | 9.7MB | 172.9ms | 83.3ms | 67.6ms | 4348ms | 2358ms | - | - |
| UNet | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/semantic_segmentation) | 65.0 | 64.28 | 52MB | 13MB | 555.4ms | 115.0ms | 82.2ms | 18687ms | 7768ms | - | - |
| Deeplabv3-ResNet50 | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/semantic_segmentation) | 79.91 | 78.78 | 150MB | 38MB | 983.2ms | 106.2ms | 82.1ms | 22515ms | 5436ms | - | - |


测试环境：
- GPU: Tesla T4; cuda11.1/cudnn8.1.1/trt8.4.0.6;
- CPU: Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz
- ARM CPU 测试环境：SDM865(4xA77+4xA55)
- PaddlePaddle 2.4.0
- CPU测试开启MKLDNN，单线程；ARM CPU单线程测试
- 以上数据使用[PaddleTest](https://github.com/PaddlePaddle/PaddleTest/tree/develop/inference/python_api_test/test_int8_model)的Python脚本测试得到

## NLP

| 模型 | 示例 | Acc<sup>FP32</sup> | Acc<sup>INT8</sup>  | 模型体积<sup>FP32</sup> | 模型体积<sup>INT8</sup> | TRT<sup>FP32</sup> | TRT<sup>FP16</sup> | TRT<sup>INT8</sup> | CPU<sup>FP32</sup> | CPU<sup>INT8</sup> | ARM-CPU<sup>FP32</sup>| ARM-CPU<sup>INT8</sup> |  
|:---:|:---:|:----:|:---:|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|:---:|:-----:|:-----:|  
| ERNIE 3.0-Medium | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/nlp) | 75.37 | 74.4 | 288MB | 155MB | 73.84ms | 11.38ms | 4.43ms | 1519ms | 591ms | - | - |
| PP-MiniLM | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/nlp) | 74.03 | 73.3 | 228MB | 106MB | 2.7ms | 73.72ms | 11.24ms | 1454ms | 702ms | - | - |
| BERT | [示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/pytorch_huggingface) | 60.3 | 58.69 | 414MB | 149MB | 13.93ms | 2.85ms | 2.13ms | 251.9ms | 79.8ms | - | - |

测试环境：
- GPU: Tesla T4; cuda11.1/cudnn8.1.1/trt8.4.0.6;
- CPU: Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz
- ARM CPU 测试环境：SDM865(4xA77+4xA55)
- PaddlePaddle 2.4.0
- CPU测试开启MKLDNN，单线程
- 以上数据使用[PaddleTest](https://github.com/PaddlePaddle/PaddleTest/tree/develop/inference/python_api_test/test_int8_model)的Python脚本测试得到
