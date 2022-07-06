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
| MobileNetV1 | 量化+蒸馏 | 70.57 | - | 13.64 |
| ResNet50_vd | Baseline | 79.12 | 3.19 | - |
| ResNet50_vd | 量化+蒸馏 | 78.74 | 0.92 | - |
| ShuffleNetV2_x1_0 | Baseline | 68.65 | - | 10.43 |
| ShuffleNetV2_x1_0 | 量化+蒸馏 | 68.32 | - | 5.51 |
| SqueezeNet1_0_infer | Baseline | 59.60 | - | 35.98 |
| SqueezeNet1_0_infer | 量化+蒸馏 | 59.45 | - | 16.96 |
| PPLCNetV2_base | Baseline | 76.86 | - | 36.50 |
| PPLCNetV2_base | 量化+蒸馏 | 76.43 | - | 15.79 |
| PPHGNet_tiny | Baseline | 79.59 | 2.82 | - |
| PPHGNet_tiny | 量化+蒸馏 | 79.20 | 0.98 | - |
| InceptionV3 | Baseline | 79.14 | 4.79 | - |
| InceptionV3 | 量化+蒸馏 | 78.32 | 1.47 | - |
| EfficientNetB0 | Baseline | 77.02 | 1.95 | - |
| EfficientNetB0 | 量化+蒸馏 | 75.39 | 1.44 | - |
| GhostNet_x1_0 | Baseline | 74.02 | 2.93 | - |
| GhostNet_x1_0 | 量化+蒸馏 | 72.62 | 1.03 | - |
| MobileNetV3_large_x1_0 | Baseline | 75.32 | - | 16.62 |
| MobileNetV3_large_x1_0 | 量化+蒸馏 | 70.93 | - | 9.85 |

- ARM CPU 测试环境：`SDM865(4xA77+4xA55)`
- Nvidia GPU 测试环境：
  - 硬件：NVIDIA Tesla T4 单卡
  - 软件：CUDA 11.2, cuDNN 8.0, TensorRT 8.4
  - 测试配置：batch_size: 1, image size: 224

## 3. 自动压缩流程

#### 3.1 准备环境

- python >= 3.6
- PaddlePaddle >= 2.3 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim >= 2.3

安装paddlepaddle：
```shell
# CPU
pip install paddlepaddle
# GPU
pip install paddlepaddle-gpu
```

安装paddleslim：
```shell
pip install paddleslim
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

蒸馏量化自动压缩示例通过run.py脚本启动，会使用接口 ```paddleslim.auto_compression.AutoCompression``` 对模型进行量化训练和蒸馏。配置config文件中模型路径、数据集路径、蒸馏、量化和训练等部分的参数，配置完成后便可开始自动压缩。

**单卡启动**

```shell
export CUDA_VISIBLE_DEVICES=0
python run.py --save_dir='./save_quant_mobilev1/' --config_path='./configs/MobileNetV1/qat_dis.yaml'
```

**分布式训练**

图像分类训练任务中往往包含大量训练数据，以ImageNet为例，ImageNet22k数据集中包含1400W张图像，如果使用单卡训练，会非常耗时，使用分布式训练可以达到几乎线性的加速比。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch run.py --save_dir='./save_quant_mobilev1/' --config_path='./configs/MobileNetV1/qat_dis.yaml'
```
多卡训练（分布式训练）指的是将训练任务按照一定方法拆分到多个训练节点完成数据读取、前向计算、反向梯度计算等过程，并将计算出的梯度上传至服务节点。服务节点在收到所有训练节点传来的梯度后，会将梯度聚合并更新参数。最后将参数发送给训练节点，开始新一轮的训练。多卡训练一轮训练能训练```batch size * num gpus```的数据，比如单卡的```batch size```为32，单轮训练的数据量即32，而四卡训练的```batch size```为32，单轮训练的数据量为128。

注意 ```learning rate``` 与 ```batch size``` 呈线性关系，这里单卡 ```batch size``` 为32，对应的 ```learning rate``` 为0.015，那么如果 ```batch size``` 减小4倍改为8，```learning rate``` 也需除以4；多卡时 ```batch size``` 为32，```learning rate``` 需乘上卡数。所以改变 ```batch size``` 或改变训练卡数都需要对应修改 ```learning rate```。


## 4.预测部署
#### 4.1 Python预测推理


准备好inference模型后，使用以下命令进行预测：
```shell
python infer.py --config_path="configs/infer.yaml"
```

在配置文件```configs/infer.yaml```中有以下字段用于配置预测参数：
- ```inference_model_dir```：inference 模型文件所在目录，该目录下需要有文件 .pdmodel 和 .pdiparams 两个文件
- ```model_filename```：inference_model_dir文件夹下的模型文件名称
- ```params_filename```：inference_model_dir文件夹下的参数文件名称
- ```batch_size```：预测一个batch的大小
- ```image_size```：输入图像的大小
- ```use_tensorrt```：是否使用 TesorRT 预测引擎
- ```use_gpu```：是否使用 GPU 预测
- ```enable_mkldnn```：是否启用```MKL-DNN```加速库，注意```enable_mkldnn```与```use_gpu```同时为```True```时，将忽略```enable_mkldnn```，而使用```GPU```预测
- ```use_fp16```：是否启用```FP16```
- ```use_int8```：是否启用```INT8```

注意：
- 请注意模型的输入数据尺寸，如InceptionV3输入尺寸为299，部分模型需要修改参数：```image_size```
- 如果希望提升评测模型速度，使用 ```GPU``` 评测时，建议开启 ```TensorRT``` 加速预测，使用 ```CPU``` 评测时，建议开启 ```MKL-DNN``` 加速预测
- 若使用 TesorRT 预测引擎，需安装 ```WITH_TRT=ON``` 的Paddle，下载地址：[Python预测库](https://paddleinference.paddlepaddle.org.cn/master/user_guides/download_lib.html#python)


#### 4.2 PaddleLite端侧部署
PaddleLite端侧部署可参考：
- [Paddle Lite部署](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/inference_deployment/paddle_lite_deploy.md)

## 5.FAQ
