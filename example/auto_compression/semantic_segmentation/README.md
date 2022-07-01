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

| 模型 | 策略  | Total IoU | ARM CPU耗时(ms)<br>thread=1 |Nvidia GPU耗时(ms)| 配置文件 | Inference模型  |
|:-----:|:-----:|:----------:|:---------:| :------:|:------:|:------:|
| PP-HumanSeg-Lite | Baseline |  92.87 | 56.363 |-| - | [model](https://paddleseg.bj.bcebos.com/dygraph/ppseg/ppseg_lite_portrait_398x224_with_softmax.tar.gz) |
| PP-HumanSeg-Lite | 非结构化稀疏+蒸馏 |  92.35 | 37.712 |-| [config](./configs/pp_human/pp_human_sparse.yaml)| - |
| PP-HumanSeg-Lite | 量化+蒸馏 |  92.84 | 49.656 |-| [config](./configs/pp_human/pp_human_qat.yaml) | - |
| PP-Liteseg | Baseline |  77.04| - | 1.425| - |[model](https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-PPLIteSegSTDC1.zip)|
| PP-Liteseg | 量化训练 |  76.93 | - | 1.158|[config](./configs/pp_liteseg/pp_liteseg_qat.yaml) | - |
| HRNet | Baseline |  78.97 | - |8.188|-| [model](https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-HRNetW18-Seg.zip)|
| HRNet | 量化训练 |  78.90 | - |5.812| [config](./configs/hrnet/hrnet_qat.yaml) | - |
| UNet | Baseline | 65.00  | - |15.291|-| [model](https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-UNet.zip) |
| UNet | 量化训练 |  64.93 | - |10.228| [config](./configs/unet/unet_qat.yaml) | - |
| Deeplabv3-ResNet50 | Baseline |  79.90 | -|12.766| -| [model](https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-Deeplabv3-ResNet50.zip)|
| Deeplabv3-ResNet50 | 量化训练 |  78.89 | - |8.839|[config](./configs/deeplabv3/deeplabv3_qat.yaml) | - |

- ARM CPU测试环境：`SDM710 2*A75(2.2GHz) 6*A55(1.7GHz)`；

- Nvidia GPU测试环境：

  - 硬件：NVIDIA Tesla T4 单卡
  - 软件：CUDA 11.0, cuDNN 8.0, TensorRT 8.0
  - 测试配置：batch_size: 40, max_seq_len: 128

- PP-HumanSeg-Lite数据集

  - 数据集：AISegment + PP-HumanSeg14K + 内部自建数据集。其中 AISegment 是开源数据集，可从[链接](https://github.com/aisegmentcn/matting_human_datasets)处获取；PP-HumanSeg14K 是 PaddleSeg 自建数据集，可从[官方渠道](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/contrib/PP-HumanSeg/paper.md#pp-humanseg14k-a-large-scale-teleconferencing-video-dataset)获取；内部数据集不对外公开。
  - 示例数据集: 用于快速跑通人像分割的压缩和推理流程, 不能用该数据集复现 benckmark 表中的压缩效果。 [下载链接](https://paddleseg.bj.bcebos.com/humanseg/data/mini_supervisely.zip)

- PP-Liteseg，HRNet，UNet，Deeplabv3-ResNet50数据集

  - cityscapes: 请从[cityscapes官网](https://www.cityscapes-dataset.com/login/)下载完整数据
  - 示例数据集: cityscapes数据集的一个子集，用于快速跑通压缩和推理流程，不能用该数据集复现 benchmark 表中的压缩效果。[下载链接](https://bj.bcebos.com/v1/paddle-slim-models/data/mini_cityscapes/mini_cityscapes.tar)

下面将以开源数据集为例介绍如何对PP-HumanSeg-Lite进行自动压缩。

## 3. 自动压缩流程

#### 3.1 准备环境

- PaddlePaddle >= 2.3 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim develop版本
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
https://github.com/PaddlePaddle/PaddleSlim.git
python setup.py install
```

安装paddleseg

```shell
pip install paddleseg
```

注：安装[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)的目的只是为了直接使用PaddleSeg中的Dataloader组件，不涉及模型组网等。

#### 3.2 准备数据集

开发者可下载开源数据集 (如[AISegment](https://github.com/aisegmentcn/matting_human_datasets)) 或自定义语义分割数据集。请参考[PaddleSeg数据准备文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/data/marker/marker_cn.md)来检查、对齐数据格式即可。

可以通过以下命令下载人像分割示例数据:

```shell
cd ./data
python download_data.py mini_humanseg

```

#### 3.3 准备预测模型

预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。

注：其他像`__model__`和`__params__`分别对应`model.pdmodel` 和 `model.pdiparams`文件。

- 如果想快速体验，可直接下载PP-HumanSeg-Lite 的预测模型：

```shell
wget https://bj.bcebos.com/v1/paddlemodels/PaddleSlim/analysis/ppseg_lite_portrait_398x224_with_softmax.tar.gz
tar -xzf ppseg_lite_portrait_398x224_with_softmax.tar.gz
```

也可进入[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) 中导出所需预测模型。

#### 3.4 自动压缩并产出模型

自动压缩示例通过run.py脚本启动，会使用接口```paddleslim.auto_compression.AutoCompression```对模型进行自动压缩。首先要配置config文件中模型路径、数据集路径、蒸馏、量化、稀疏化和训练等部分的参数，配置完成后便可对模型进行非结构化稀疏、蒸馏和量化、蒸馏。

当只设置训练参数，并传入``deploy_hardware``字段时，将自动搜索压缩策略进行压缩。以骁龙710（SD710）为部署硬件，进行自动压缩的运行命令如下：

```shell
# 单卡启动
export CUDA_VISIBLE_DEVICES=0
python run.py \
    --model_dir='./ppseg_lite_portrait_398x224_with_softmax' \
    --model_filename='model.pdmodel' \
    --params_filename='model.pdiparams' \
    --save_dir='./save_model' \
    --strategy_config='configs/pp_humanseg/pp_humanseg_auto.yaml' \
    --dataset_config='configs/dataset/humanseg_dataset.yaml' \
    --deploy_hardware='SD710'

# 多卡启动
export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch run.py \
    --model_dir='./ppseg_lite_portrait_398x224_with_softmax' \
    --model_filename='model.pdmodel' \
    --params_filename='model.pdiparams' \
    --save_dir='./save_model' \
    --strategy_config='configs/pp_humanseg/pp_humanseg_auto.yaml' \
    --dataset_config='configs/dataset/humanseg_dataset.yaml' \
    --deploy_hardware='SD710'
```
- 自行配置稀疏参数进行非结构化稀疏和蒸馏训练，配置参数含义详见[自动压缩超参文档](https://github.com/PaddlePaddle/PaddleSlim/blob/27dafe1c722476f1b16879f7045e9215b6f37559/demo/auto_compression/hyperparameter_tutorial.md)。具体命令如下所示：
```shell
# 单卡启动
export CUDA_VISIBLE_DEVICES=0
python run.py \
    --model_dir='./ppseg_lite_portrait_398x224_with_softmax' \
    --model_filename='model.pdmodel' \
    --params_filename='model.pdiparams' \
    --save_dir='./save_model' \
    --strategy_config='configs/pp_humanseg/pp_humanseg_sparse.yaml' \
    --dataset_config='configs/dataset/humanseg_dataset.yaml'

# 多卡启动
export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch run.py \
    --model_dir='./ppseg_lite_portrait_398x224_with_softmax' \
    --model_filename='model.pdmodel' \
    --params_filename='model.pdiparams' \
    --save_dir='./save_model' \
    --strategy_config='configs/pp_humanseg/pp_humanseg_sparse.yaml' \
    --dataset_config='configs/dataset/humanseg_dataset.yaml'
```

- 自行配置量化参数进行量化和蒸馏训练，配置参数含义详见[自动压缩超参文档](https://github.com/PaddlePaddle/PaddleSlim/blob/27dafe1c722476f1b16879f7045e9215b6f37559/demo/auto_compression/hyperparameter_tutorial.md)。具体命令如下所示：
```shell
# 单卡启动
export CUDA_VISIBLE_DEVICES=0
python run.py \
    --model_dir='./ppseg_lite_portrait_398x224_with_softmax' \
    --model_filename='model.pdmodel' \
    --params_filename='model.pdiparams' \
    --save_dir='./save_model' \
    --strategy_config='configs/pp_humanseg/pp_humanseg_qat.yaml' \
    --dataset_config='configs/dataset/humanseg_dataset.yaml'

# 多卡启动
export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch run.py \
    --model_dir='./ppseg_lite_portrait_398x224_with_softmax' \
    --model_filename='model.pdmodel' \
    --params_filename='model.pdiparams' \
    --save_dir='./save_model' \
    --strategy_config='configs/pp_humanseg/pp_humanseg_qat.yaml' \
    --dataset_config='configs/dataset/humanseg_dataset.yaml'
```

压缩完成后会在`save_dir`中产出压缩好的预测模型，可直接预测部署。


## 4.预测部署

- [Paddle Inference Python部署](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/deployment/inference/python_inference.md)
- [Paddle Inference C++部署](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/deployment/inference/cpp_inference.md)
- [Paddle Lite部署](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/deployment/lite/lite.md)

## 5.FAQ
