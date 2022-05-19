# 语义分割模型自动压缩示例

目录：
- [1.简介](#1简介)
- [2.Benchmark](#2Benchmark)
- [3.开始自动压缩](#自动压缩流程)
  - [3.1 环境准备](#31-准备环境)
  - [3.2 准备数据集](#32-准备数据集)
  - [3.3 准备预测模型](#33-准备预测模型)
  - [3.4 自动压缩并产出模型](#34-自动压缩并产出模型)
- [4.预测部署](#4预测部署)
- [5.FAQ](5FAQ)

## 1.简介

本示例将以语义分割模型PP-HumanSeg-Lite为例，介绍如何使用PaddleSeg中Inference部署模型进行自动压缩。本示例使用的自动压缩策略为非结构化稀疏、蒸馏和量化、蒸馏。

## 2.Benchmark

- [PP-HumanSeg-Lite](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/contrib/PP-HumanSeg#portrait-segmentation)

| 模型 | 策略  | Total IoU | 耗时(ms)<br>thread=1 | 配置文件 | Inference模型  |
|:-----:|:-----:|:----------:|:---------:| :------:| :------:|
| PP-HumanSeg-Lite | Baseline |  0.9287 | 56.363 | - | [model](https://paddleseg.bj.bcebos.com/dygraph/ppseg/ppseg_lite_portrait_398x224_with_softmax.tar.gz) |
| PP-HumanSeg-Lite | 非结构化稀疏+蒸馏 |  0.9235 | 37.712 | [config](./configs/pp_human_sparse_dis.yaml)| - |
| PP-HumanSeg-Lite | 量化+蒸馏 |  0.9284 | 49.656 | [config](./configs/pp_human_sparse_dis.yaml) | - |

- 测试环境：`SDM710 2*A75(2.2GHz) 6*A55(1.7GHz)`；
- 测试数据集：AISegment + PP-HumanSeg14K + 内部自建数据集。

下面将以开源数据集为例介绍如何进行自动压缩。

## 3. 自动压缩流程

#### 3.1 准备环境

- PaddlePaddle >= 2.2 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim >= 2.3 或者适当develop版本
- PaddleSeg >= 2.5

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

安装paddleseg

```shell
pip install paddleseg
```

注：安装[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)的目的只是为了直接使用PaddleSeg中的Dataloader组件，不涉及模型组网等。

#### 3.2 准备数据集

开发者可下载开源数据集或自定义语义分割数据集，例如PP-HumanSeg-Lite模型中使用的语义分割数据集[PP-HumanSeg14K](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/contrib/PP-HumanSeg/paper.md#pp-humanseg14k-a-large-scale-teleconferencing-video-dataset)可从官方渠道下载。

如果是自定义数据，请参考[PaddleSeg数据准备文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/data/marker/marker_cn.md)来检查对齐数据格式即可。

#### 3.3 准备预测模型

预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。

注：其他像`__model__`和`__params__`分别对应`model.pdmodel` 和 `model.pdiparams`文件。

- 如果想快速体验，可直接下载PP-HumanSeg-Lite 的预测模型：

```shell
wget https://paddleseg.bj.bcebos.com/dygraph/ppseg/ppseg_lite_portrait_398x224_with_softmax.tar.gz
tar -xzf ppseg_lite_portrait_398x224_with_softmax.tar.gz
```

也可进入[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) 中导出所需预测模型。

#### 3.4 自动压缩并产出模型

自动压缩示例通过run.py脚本启动，会使用接口```paddleslim.auto_compression.AutoCompression```对模型进行自动压缩。首先要配置config文件中模型路径、数据集路径、蒸馏、量化、稀疏化和训练等部分的参数，配置完成后便可对模型进行非结构化稀疏、蒸馏和量化、蒸馏。

当只设置训练参数，并传入``deploy_hardware``字段时，将自动搜索压缩策略进行压缩。以骁龙710（SD710）为部署硬件，进行自动压缩的运行命令如下：

```shell
export CUDA_VISIBLE_DEVICES=0
python run.py \
    --model_dir='./ppseg_lite_portrait_398x224_with_softmax' \
    --model_filename='model.pdmodel' \
    --params_filename='model.pdiparams' \
    --save_dir='./save_model' \
    --config_path='configs/pp_humanseg_auto.yaml' \
    --deploy_hardware='SD710'
```
- 自行配置稀疏参数进行非结构化稀疏和蒸馏训练，配置参数含义详见[自动压缩超参文档](https://github.com/PaddlePaddle/PaddleSlim/blob/27dafe1c722476f1b16879f7045e9215b6f37559/demo/auto_compression/hyperparameter_tutorial.md)。具体命令如下所示：
```shell
export CUDA_VISIBLE_DEVICES=0
python run.py \
    --model_dir='./ppseg_lite_portrait_398x224_with_softmax' \
    --model_filename='model.pdmodel' \
    --params_filename='model.pdiparams' \
    --save_dir='./save_model' \
    --config_path='configs/pp_humanseg_sparse_dis.yaml'
```

- 自行配置量化参数进行量化和蒸馏训练，配置参数含义详见[自动压缩超参文档](https://github.com/PaddlePaddle/PaddleSlim/blob/27dafe1c722476f1b16879f7045e9215b6f37559/demo/auto_compression/hyperparameter_tutorial.md)。具体命令如下所示：
```shell
export CUDA_VISIBLE_DEVICES=0
python run.py \
    --model_dir='./ppseg_lite_portrait_398x224_with_softmax' \
    --model_filename='model.pdmodel' \
    --params_filename='model.pdiparams' \
    --save_dir='./save_model' \
    --config_path='configs/pp_humanseg_quant_dis.yaml'
```

压缩完成后会在`save_dir`中产出压缩好的预测模型，可直接预测部署。


## 4.预测部署

- [Paddle Inference Python部署](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/deployment/inference/python_inference.md)
- [Paddle Inference C++部署](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/deployment/inference/cpp_inference.md)
- [Paddle Lite部署](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/deployment/lite/lite.md)

## 5.FAQ
