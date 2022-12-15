# 图像分类模型全量化示例

目录：
- [1. 简介](#1简介)
- [2. Benchmark](#2Benchmark)
- [3. 全量化流程](#全量化流程)
  - [3.1 准备环境](#31-准备准备)
  - [3.2 准备数据集](#32-准备数据集)
  - [3.3 准备预测模型](#33-准备预测模型)
  - [3.4 全量化并产出模型](#34-全量化并产出模型)
- [4. 预测部署](#4预测部署)
  - [4.1 PaddleLite端侧部署](#42-PaddleLite端侧部署)
- [5. FAQ](5FAQ)


## 1. 简介
本示例将以图像分类模型MobileNetV1为例，介绍如何使用PaddleClas中Inference部署模型进行全量化。本示例全量化的策略使用了量化训练和蒸馏。

## 2. Benchmark

### PaddleClas模型

| 模型 | 策略 | Top-1 Acc | GPU 耗时(ms) | ARM CPU 耗时(ms) | 配置文件 | Inference模型 |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| MobileNetV3_large_x1_0 | Baseline | 75.32 | - | - | - | [Model](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar) |
| MobileNetV3_large_x1_0 | 全量化 | 74.41 | - | - | [Config](./configs/mobilenetv3_large_qat_dis.yaml) | [Model](https://paddle-slim-models.bj.bcebos.com/act/MobileNetV3_large_x1_0_QAT.tar) |


## 3. 全量化流程

#### 3.1 准备环境

- python >= 3.6
- PaddlePaddle >= 2.4 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim >= 2.4

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
本案例默认以ImageNet1k数据进行全量化实验，如数据集为非ImageNet1k格式数据， 请参考[PaddleClas数据准备文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/data_preparation/classification_dataset.md)。将下载好的数据集放在当前目录下`./ILSVRC2012`。


#### 3.3 准备预测模型
预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。

注：其他像`__model__`和`__params__`分别对应`model.pdmodel` 和 `model.pdiparams`文件。

可在[PaddleClas预训练模型库](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/algorithm_introduction/ImageNet_models.md)中直接获取Inference模型，具体可参考下方获取MobileNetV1模型示例：

```shell
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar
tar -xf MobileNetV3_large_x1_0_infer.tar
```
也可根据[PaddleClas文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/inference_deployment/export_model.md)导出Inference模型。

#### 3.4 全量化并产出模型

全量化示例通过run.py脚本启动，会使用接口 ```paddleslim.auto_compression.AutoCompression``` 对模型进行量化训练和蒸馏。配置config文件中模型路径、数据集路径、蒸馏、量化和训练等部分的参数，配置完成后便可开始全量化。

**单卡启动**

```shell
export CUDA_VISIBLE_DEVICES=0
python run.py --save_dir='./save_quant_mobilev3/' --config_path='./configs/mobilenetv3_large_qat_dis.yaml'
```

**多卡启动**

图像分类训练任务中往往包含大量训练数据，以ImageNet为例，ImageNet22k数据集中包含1400W张图像，如果使用单卡训练，会非常耗时，使用分布式训练可以达到几乎线性的加速比。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch run.py --save_dir='./save_quant_mobilev3/' --config_path='./configs/mobilenetv3_large_qat_dis.yaml'
```
多卡训练指的是将训练任务按照一定方法拆分到多个训练节点完成数据读取、前向计算、反向梯度计算等过程，并将计算出的梯度上传至服务节点。服务节点在收到所有训练节点传来的梯度后，会将梯度聚合并更新参数。最后将参数发送给训练节点，开始新一轮的训练。多卡训练一轮训练能训练```batch size * num gpus```的数据，比如单卡的```batch size```为32，单轮训练的数据量即32，而四卡训练的```batch size```为32，单轮训练的数据量为128。

注意 ```learning rate``` 与 ```batch size``` 呈线性关系，这里单卡 ```batch size``` 为32，对应的 ```learning rate``` 为0.015，那么如果 ```batch size``` 减小4倍改为8，```learning rate``` 也需除以4；多卡时 ```batch size``` 为32，```learning rate``` 需乘上卡数。所以改变 ```batch size``` 或改变训练卡数都需要对应修改 ```learning rate```。

**验证精度**

根据训练log可以看到模型验证的精度，若需再次验证精度，修改配置文件```./configs/mobilenetv3_large_qat_dis.yaml```中所需验证模型的文件夹路径及模型和参数名称```model_dir, model_filename, params_filename```，然后使用以下命令进行验证：

```shell
export CUDA_VISIBLE_DEVICES=0
python eval.py --config_path='./configs/mobilenetv3_large_qat_dis.yaml'
```


## 4.预测部署

#### 4.1 PaddleLite端侧部署
PaddleLite端侧部署可参考：
- [Paddle Lite全量化模型部署](https://www.paddlepaddle.org.cn/lite/develop/demo_guides/verisilicon_timvx.html#tim-vx)

## 5.FAQ
