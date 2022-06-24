# 图像分类模型自动压缩示例

目录：
- [1. 简介](#1简介)
- [2. Benchmark](#2Benchmark)
- [3. 自动压缩流程](#自动压缩流程)
  - [3.1 准备环境](#31-准备准备)
  - [3.2 准备数据集](#32-准备数据集)
  - [3.3 准备预测模型](#33-准备预测模型)
  - [3.4 自动压缩并产出模型](#34-自动压缩并产出模型)
- [4. 预测部署](#4预测部署)
- [5. FAQ](5FAQ)


## 1. 简介
本示例将以图像分类模型MobileNetV1为例，介绍如何使用PaddleClas中Inference部署模型进行自动压缩。本示例使用的自动压缩策略为量化训练和蒸馏。

## 2. Benchmark
- PaddleClas模型

| 模型 | 策略 | Top-1 Acc | GPU 耗时(ms) | ARM CPU 耗时(ms) ｜ 
|:------:|:------:|:------:|:------:|:------:|
| MobileNetV1 | Baseline | 70.90 | - | 33.15 |
| MobileNetV1 | 量化+蒸馏 | 70.49 | - | 13.64 |
| ResNet50_vd | Baseline | 79.12 | 3.1901 | - |
| ResNet50_vd | 量化+蒸馏 | 78.55 | 0.9243 | - |
| ShuffleNetV2_x1_0 | Baseline | 68.65 | - | 10.43 |
| ShuffleNetV2_x1_0 | 量化+蒸馏 | 67.78 | - | 5.51 |
| SqueezeNet1_0_infer | Baseline | 59.6 | - | 35.98 |
| SqueezeNet1_0_infer | 量化+蒸馏 | 59.13 | - | 16.96 |
| PPLCNetV2_base | Baseline | 76.86 | - | 36.50 |
| PPLCNetV2_base | 量化+蒸馏 | 76.43 | - | 15.79 |
| PPHGNet_tiny | Baseline | 79.59 | 2.822 | - |
| PPHGNet_tiny | 量化+蒸馏 | 79.19 |  | - |
| EfficientNetB0 | Baseline | 77.02 | 1.9529 | - |
| EfficientNetB0 | 量化+蒸馏 | 73.61 | 1.4362 | - |
| GhostNet_x1_0 | Baseline | 74.02 | 2.9301 | - |
| GhostNet_x1_0 | 量化+蒸馏 | 71.11 | 1.0348 | - |
| InceptionV3 | Baseline | 79.14 | 4.7931 | - |
| InceptionV3 | 量化+蒸馏 | 73.16 | 1.4703 | - |
| MobileNetV3_large_x1_0 | Baseline | 75.32 | - | 16.62 |
| MobileNetV3_large_x1_0 | 量化+蒸馏 | 68.84 | - | 9.85 |

- ARM CPU 测试环境：`SDM865(4xA77+4xA55)`
- Nvidia GPU 测试环境：
  - 硬件：NVIDIA Tesla T4 单卡
  - 软件：CUDA 11.2, cuDNN 8.0, TensorRT 8.4
  - 测试配置：batch_size: 1, image size: 224



## 3. 自动压缩流程

#### 3.1 准备环境

- python >= 3.6
- PaddlePaddle >= 2.3 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim develop版本

安装paddlepaddle：
```shell
# CPU
pip install paddlepaddle
# GPU
pip install paddlepaddle-gpu
```

安装paddleslim：
```shell
https://github.com/PaddlePaddle/PaddleSlim.git
python setup.py install
```

#### 3.2 准备数据集
本案例默认以ImageNet1k数据进行自动压缩实验，如数据集为非ImageNet1k格式数据， 请参考[PaddleClas数据准备文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/data_preparation/classification_dataset.md)。


#### 3.3 准备预测模型
预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。

注：其他像`__model__`和`__params__`分别对应`model.pdmodel` 和 `model.pdiparams`文件。

可在[PaddleClas预训练模型库](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/algorithm_introduction/ImageNet_models.md)中直接获取Inference模型，具体可参考下方获取MobileNetV1模型示例：

```shell
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar
tar -xf MobileNetV1_infer.tar
```
也可根据[PaddleClas文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/inference_deployment/export_model.md)导出Inference模型。

#### 3.4 自动压缩并产出模型

蒸馏量化自动压缩示例通过run.py脚本启动，会使用接口```paddleslim.auto_compression.AutoCompression```对模型进行量化训练和蒸馏。配置config文件中模型路径、数据集路径、蒸馏、量化和训练等部分的参数，配置完成后便可开始自动压缩。

```shell
# 单卡启动
export CUDA_VISIBLE_DEVICES=0
# 多卡启动
export CUDA_VISIBLE_DEVICES=0,1,2,3

python run.py --save_dir='./save_quant_mobilev1/' --config_path='./configs/MobileNetV1/qat_dis.yaml'

```


## 4.预测部署

- [Paddle Inference Python部署](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/deployment/inference/python_inference.md)
- [Paddle Inference C++部署](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/deployment/inference/cpp_inference.md)
- [Paddle Lite部署](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/deployment/lite/lite.md)

## 5.FAQ
