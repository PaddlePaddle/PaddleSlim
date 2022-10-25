# TensorFlow图像分类模型自动压缩示例

目录：
- [1. 简介](#1简介)
- [2. Benchmark](#2Benchmark)
- [3. 自动压缩流程](#自动压缩流程)
  - [3.1 准备环境](#31-准备准备)
  - [3.2 准备数据集](#32-准备数据集)
  - [3.3 X2Paddle转换模型流程](#33-X2Paddle转换模型流程)
  - [3.4 自动压缩并产出模型](#34-自动压缩并产出模型)
- [4. 预测部署](#4预测部署)
- [5. FAQ](5FAQ)


## 1. 简介
飞桨模型转换工具[X2Paddle](https://github.com/PaddlePaddle/X2Paddle)支持将```Caffe/TensorFlow/ONNX/PyTorch```的模型一键转为飞桨（PaddlePaddle）的预测模型。借助X2Paddle的能力，PaddleSlim的自动压缩功能可方便地用于各种框架的推理模型。

本示例将以[TensorFlow](https://github.com/tensorflow/tensorflow)框架的MobileNetV1模型为例，介绍如何自动压缩其他框架中的图像分类模型。本示例会利用[TensorFlow](https://github.com/tensorflow/models)开源models库，将TensorFlow框架模型转换为Paddle框架模型，再使用ACT自动压缩功能进行自动压缩。本示例使用的自动压缩策略为量化训练。

## 2. Benchmark
| 模型 | 策略 | Top-1 Acc | 耗时(ms) threads=1 | Inference模型 |
|:------:|:------:|:------:|:------:|:------:|
| MobileNetV1 | Base模型 | 71.0 | 30.45 | [Model](https://paddle-slim-models.bj.bcebos.com/act/mobilenetv1_inference_model_tf2paddle.tar) |
| MobileNetV1 | 量化+蒸馏 | 70.22 | 15.86 | [Model](https://paddle-slim-models.bj.bcebos.com/act/mobilenetv1_quant.tar) |

- 测试环境：`骁龙865 4*A77 4*A55`

说明：
- MobileNetV1模型源自[tensorflow/models](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)

## 3. 自动压缩流程

#### 3.1 准备环境
- PaddlePaddle >= 2.4rc0 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim >= 2.4rc
- [X2Paddle](https://github.com/PaddlePaddle/X2Paddle) >= 1.3.6
- opencv-python

（1）安装paddlepaddle：
```shell
# CPU
pip install paddlepaddle==2.4rc0
# GPU
pip install paddlepaddle_gpu==2.4rc0
```

（2）安装paddleslim：
```shell
pip install paddleslim==2.4rc
```

（3）安装TensorFlow:
```shell
pip install tensorflow == 1.14
```

（3）安装X2Paddle的1.3.6以上版本：
```shell
pip install x2paddle
```

#### 3.2 准备数据集
本案例默认以ImageNet1k数据进行自动压缩实验。

#### 3.3 准备预测模型

（1）转换模型

```
x2paddle --framework=tensorflow --model=tf_model.pb --save_dir=pd_model
```
即可得到MobileNetV1模型的预测模型（`model.pdmodel` 和 `model.pdiparams`）。如想快速体验，可直接下载上方表格中MobileNetV1的[Base模型](https://paddle-slim-models.bj.bcebos.com/act/mobilenetv1_inference_model_tf2paddle.tar)。

预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。

### 3.4 自动压缩并产出模型

蒸馏量化自动压缩示例通过run.py脚本启动，会使用接口```paddleslim.auto_compression.AutoCompression```对模型进行自动压缩。配置config文件中模型路径、蒸馏、量化、和训练等部分的参数，配置完成后便可对模型进行量化和蒸馏。具体运行命令为：
```
# 单卡
export CUDA_VISIBLE_DEVICES=0
python run.py --config_path=./configs/mbv1_qat_dis.yaml --save_dir='./output/'
```

#### 3.5 测试模型精度

使用eval.py脚本得到模型的mAP：
```
export CUDA_VISIBLE_DEVICES=0
python eval.py --config_path=./configs/mbv1_qat_dis.yaml
```

## 4.预测部署

#### 4.1 PaddleLite端侧部署
PaddleLite端侧部署可参考：
- [Paddle Lite部署](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/inference_deployment/paddle_lite_deploy.md)

## 5.FAQ
