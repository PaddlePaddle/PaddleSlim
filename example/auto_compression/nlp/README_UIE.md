# 自然语言处理模型自动压缩示例

目录：
- [1. 简介](#1简介)
- [2. Benchmark](#2Benchmark)
- [3. 自动压缩流程](#自动压缩流程)
  - [3.1 准备环境](#31-准备准备)
  - [3.2 准备数据集](#32-准备数据集)
  - [3.3 准备预测模型](#33-准备预测模型)
  - [3.4 自动压缩并产出模型](#34-自动压缩并产出模型)
- [4. 压缩配置介绍](#4压缩配置介绍)
- [5. 预测部署](#5预测部署)
- [6. FAQ](6FAQ)


## 1. 简介
本示例将以自然语言处理模型UIE-base为例，介绍如何使用PaddleNLP中UIE Inference部署模型进行自动压缩.

## 2. Benchmark


模型精度对比如下：
| 模型 | 策略 | |
|:------:|:------:|:------:|

模型在不同任务上平均精度以及加速对比如下：
|  模型 |策略| Accuracy（avg） | 时延(ms) | 加速比 |
|:-------:|:--------:|:----------:|:------------:| :------:|

性能测试的环境为
- 硬件：NVIDIA Tesla T4 单卡
- 软件：CUDA 11.0, cuDNN 8.0, TensorRT 8.0
- 测试配置：batch_size: 40, max_seq_len: 128

## 3. 自动压缩流程

#### 3.1 准备环境
- python >= 3.6
- PaddlePaddle >= 2.3 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim >= 2.3
- PaddleNLP >= 2.3

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

安装paddlenlp：
```shell
pip install paddlenlp
```

注：安装PaddleNLP的目的是为了下载PaddleNLP中的数据集和Tokenizer。

#### 3.2 准备数据集
本案例默认以CLUE数据进行自动压缩实验，如数据集为非CLUE格式数据，请修改启动文本run.sh中dataset字段，PaddleNLP会自动下载对应数据集。


#### 3.3 准备预测模型
预测模型的格式为：`model.pdmodel` 和 `model.pdiparams`两个，带`pdmodel`的是模型文件，带`pdiparams`后缀的是权重文件。

注：其他像`__model__`和`__params__`分别对应`model.pdmodel` 和 `model.pdiparams`文件。

##### 直接下载已微调模型

| 模型 |
|:------:|

从上表获得模型超链接, 并用以下命令下载推理模型文件:

```shell
```

##### 重新微调模型


#### 3.4 自动压缩并产出模型

自动压缩示例通过run_uie.py脚本启动，会使用接口```paddleslim.auto_compression.AutoCompression```对模型进行自动压缩。配置config文件中训练部分的参数，将任务名称、模型类型、数据集名称、压缩参数传入，配置完成后便可对模型进行剪枝、蒸馏训练和离线量化。
数据集为CLUE，不同任务名称代表CLUE上不同的任务，可选择的任务名称有：```afqmc, tnews, iflytek, ocnli, cmnli, cluewsc2020, csl```。具体运行命令为：
：
```shell
export CUDA_VISIBLE_DEVICES=0
python run_uie.py --config_path='./configs/uie.yaml' --save_dir='./save_uie/'
```


## 4. 压缩配置介绍
自动压缩需要准备config文件，并传入```config_path```字段，configs文件夹下可查看不同任务的配置文件，以下示例以afqmc数据集为例介绍。训练参数需要自行配置。蒸馏、剪枝和离线量化的相关配置，自动压缩策略可以自动获取得到，也可以自行配置。PaddleNLP模型的自动压缩实验默认使用剪枝、蒸馏和离线量化的策略。

- 训练参数

训练参数主要设置学习率、训练轮数（epochs）和优化器等。```origin_metric```是原模型精度，如设置该参数，压缩之前会先验证模型精度是否正常。

```yaml
TrainConfig:
  epochs: 6
  eval_iter: 1070
  learning_rate: 2.0e-5
  optimizer_builder:
    optimizer:
      type: AdamW
    weight_decay: 0.01
  origin_metric: 0.7403
```

以下是默认的蒸馏、剪枝和离线量化的配置：

- 蒸馏参数

蒸馏参数包括teacher网络模型路径（即微调后未剪枝的模型），自动压缩策略会自动查找教师网络节点和对应的学生网络节点进行蒸馏，不需要手动设置。

```yaml
Distillation:
  teacher_model_dir: ./uie-base
  teacher_model_filename: inference.pdmodel
  teacher_params_filename: inference.pdiparams
```

- 量化参数

量化参数主要设置量化比特数和量化op类型，其中量化op包含卷积层（conv2d, depthwise_conv2d）和全连接层（mul，matmul_v2）。

```yaml
Quantization:
  activation_bits: 8
  quantize_op_types:
  - conv2d
  - depthwise_conv2d
  - mul
  - matmul_v2
  weight_bits: 8
```

## 5. 预测部署


## 6. FAQ
