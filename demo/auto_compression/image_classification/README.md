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
  - [4.1 Python预测推理](#41-Python预测推理)
  - [4.2 PaddleLite端侧部署](#42-PaddleLite端侧部署)
- [5. FAQ](5FAQ)


## 1. 简介
本示例将以图像分类模型MobileNetV1为例，介绍如何使用PaddleClas中Inference部署模型进行自动压缩。本示例使用的自动压缩策略为量化训练和蒸馏。

## 2. Benchmark

### PaddleClas模型

| 模型 | 策略 | Top-1 Acc | GPU 耗时(ms) | ARM CPU 耗时(ms) |
|:------:|:------:|:------:|:------:|:------:|
| MobileNetV1 | Baseline | 70.90 | - | 33.15 |
| MobileNetV1 | 量化+蒸馏 | 70.49 | - | 13.64 |
| ResNet50_vd | Baseline | 79.12 | 3.19 | - |
| ResNet50_vd | 量化+蒸馏 | 78.55 | 0.92 | - |
| ShuffleNetV2_x1_0 | Baseline | 68.65 | - | 10.43 |
| ShuffleNetV2_x1_0 | 量化+蒸馏 | 67.78 | - | 5.51 |
| SqueezeNet1_0_infer | Baseline | 59.60 | - | 35.98 |
| SqueezeNet1_0_infer | 量化+蒸馏 | 59.13 | - | 16.96 |
| PPLCNetV2_base | Baseline | 76.86 | - | 36.50 |
| PPLCNetV2_base | 量化+蒸馏 | 76.43 | - | 15.79 |
| PPHGNet_tiny | Baseline | 79.59 | 2.82 | - |
| PPHGNet_tiny | 量化+蒸馏 | 79.19 | 0.98 | - |
| EfficientNetB0 | Baseline | 77.02 | 1.95 | - |
| EfficientNetB0 | 量化+蒸馏 | 73.61 | 1.44 | - |
| GhostNet_x1_0 | Baseline | 74.02 | 2.93 | - |
| GhostNet_x1_0 | 量化+蒸馏 | 71.11 | 1.03 | - |
| InceptionV3 | Baseline | 79.14 | 4.79 | - |
| InceptionV3 | 量化+蒸馏 | 73.16 | 1.47 | - |
| MobileNetV3_large_x1_0 | Baseline | 75.32 | - | 16.62 |
| MobileNetV3_large_x1_0 | 量化+蒸馏 | 68.84 | - | 9.85 |

- ARM CPU 测试环境：`SDM865(4xA77+4xA55)`
- Nvidia GPU 测试环境：
  - 硬件：NVIDIA Tesla T4 单卡
  - 软件：CUDA 11.2, cuDNN 8.0, TensorRT 8.4
  - 测试配置：batch_size: 1, image size: 224


### TensorFlow MobileNetV1模型

| 模型 | 策略 | Top-1 Acc | 耗时(ms) threads=1 | Inference模型 |
|:------:|:------:|:------:|:------:|:------:|
| MobileNetV1 | Base模型 | 71.0 | 30.45 | [Model](https://paddle-slim-models.bj.bcebos.com/act/mobilenetv1_inference_model_tf2paddle.tar) |
| MobileNetV1 | 量化+蒸馏 | 70.22 | 15.86 | [Model](https://paddle-slim-models.bj.bcebos.com/act/mobilenetv1_quant.tar) |

- 测试环境：`骁龙865 4*A77 4*A55`

说明：
- MobileNetV1模型源自[tensorflow/models](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)


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

** 单卡启动 **
```shell
export CUDA_VISIBLE_DEVICES=0
python run.py --save_dir='./save_quant_mobilev1/' --config_path='./configs/MobileNetV1/qat_dis.yaml'
```

** 分布式训练 **
图像分类训练任务中往往包含大量训练数据，以ImageNet为例，ImageNet22k数据集中包含1400W张图像，如果使用单卡训练，会非常耗时，使用分布式训练可以达到几乎线性的加速比。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch run.py --save_dir='./save_quant_mobilev1/' --config_path='./configs/MobileNetV1/qat_dis.yaml'
```
多卡训练（分布式训练）指的是将训练任务按照一定方法拆分到多个训练节点完成数据读取、前向计算、反向梯度计算等过程，并将计算出的梯度上传至服务节点。服务节点在收到所有训练节点传来的梯度后，会将梯度聚合并更新参数。最后将参数发送给训练节点，开始新一轮的训练。多卡训练一轮训练能训练```batch size * num gpus```的数据，比如单卡的```batch size```为32，单轮训练的数据量即32，而四卡训练的```batch size```为32，单轮训练的数据量为128。

注意```learning rate```与```batch size```呈线性关系，这里单卡```batch size```为32，```learning rate```为0.015，多卡时```batch size```为32，```learning rate```需乘上卡数。若改变```batch size```也需要对应修改```learning rate```。


## 4.预测部署
#### 4.1 Python预测推理
准备好inference模型后，使用以下命令进行预测：
```shell
python infer.py -c configs/infer.yaml
```

在配置文件```configs/infer.yaml```中有以下字段用于配置预测参数：
- ```Global.infer_imgs```：待预测的图片文件路径
- ```Global.inference_model_dir```：inference 模型文件所在目录，该目录下需要有文件 .pdmodel 和 .pdiparams 两个文件
- ```Global.use_tensorrt```：是否使用 TesorRT 预测引擎
- ```Global.use_gpu```：是否使用 GPU 预测
- ```Global.enable_mkldnn```：是否启用```MKL-DNN```加速库，注意```enable_mkldnn```与```use_gpu```同时为```True```时，将忽略```enable_mkldnn```，而使用```GPU```预测
- ```Global.use_fp16```：是否启用```FP16```
- ```PreProcess```：用于数据预处理配置
- ```PostProcess```：由于后处理配置
- ```PostProcess.Topk.class_id_map_file```：数据集 label 的映射文件，默认为```./images/imagenet1k_label_list.txt```，该文件为 PaddleClas 所使用的 ImageNet 数据集 label 映射文件

注意：
- 请注意模型的输入数据尺寸，部分模型需要修改参数：```PreProcess.resize_short```, ```PreProcess.resize```
- 如果希望提升评测模型速度，使用```GPU```评测时，建议开启```TensorRT```加速预测，使用```CPU```评测时，建议开启```MKL-DNN```加速预测。

#### 4.2 PaddleLite端侧部署
PaddleLite端侧部署可参考：
- [Paddle Lite部署](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/inference_deployment/paddle_lite_deploy.md)

## 5.FAQ
