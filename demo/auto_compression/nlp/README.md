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
本示例将以自然语言处理模型PP-MiniLM为例，介绍如何使用PaddleNLP中Inference部署模型进行自动压缩。本示例使用的自动压缩策略为剪枝蒸馏和离线量化(Post-training quantization)。

## 2. Benchmark
- PP-MiniLM模型

PP-MiniLM是一个6层的预训练中文小模型，使用PaddleNLP中```from_pretrained```导入PP-MiniLM之后，就可以在自己的数据集上进行fine-tuning，具体介绍可参考[PP-MiniLM文档](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/model_compression/pp-minilm#PP-MiniLM%E4%B8%AD%E6%96%87%E5%B0%8F%E6%A8%A1%E5%9E%8B)。

模型精度对比如下：
| 模型 | 策略 | AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUEWSC2020 | CSL | AVG |
|:------:|:------:|:------:|:------:|:------:|:------:|:-----------:|:------:|:------:|:------:|
| PP-MiniLM | Base模型| 74.03 | 56.66 | 60.21 | 80.98 | 76.20 | 84.21 | 77.36 | 72.81 |
| PP-MiniLM |剪枝蒸馏+离线量化| 73.56 | 56.38 | 59.87 | 80.80 | 76.44 | 82.23 | 77.77 | 72.44 |

模型在不同任务上平均精度以及加速对比如下：
|  PP-MiniLM | Accuracy（avg） | 时延(ms) | 加速比 |
|:-------:|:----------:|:------------:| :------:|
| 压缩前 |  72.81 | 128.01 | - |
| 压缩后 |  72.44 | 17.97 | 612% |

性能测试的环境为
- 硬件：NVIDIA Tesla T4 单卡
- 软件：CUDA 11.0, cuDNN 8.0, TensorRT 8.0
- 测试配置：batch_size: 40, max_seq_len: 128

## 3. 自动压缩流程

#### 3.1 准备环境
- python >= 3.6
- PaddlePaddle >= 2.2 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim >= 2.3 或者适当develop版本
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

本示例可参考[PaddleNLP PP-MiniLM 中文小模型](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/model_compression/pp-minilm)微调后保存下每个数据集下有最高准确率的模型，也可直接下载以下已微调完成的Inference模型：[afqmc](https://bj.bcebos.com/v1/paddle-slim-models/act/afqmc.tar), [tnews](https://bj.bcebos.com/v1/paddle-slim-models/act/tnews.tar), [iflytek](https://bj.bcebos.com/v1/paddle-slim-models/act/iflytek.tar),[ ocnli](https://bj.bcebos.com/v1/paddle-slim-models/act/ocnli.tar), [cmnli](https://bj.bcebos.com/v1/paddle-slim-models/act/cmnli.tar), [cluewsc2020](https://bj.bcebos.com/v1/paddle-slim-models/act/cluewsc.tar), [csl](https://bj.bcebos.com/v1/paddle-slim-models/act/csl.tar)。其他模型可根据[PaddleNLP文档](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples)导出Inference模型。

```shell
wget https://bj.bcebos.com/v1/paddle-slim-models/act/afqmc.tar
tar -zxvf afqmc.tar
```

#### 3.4 自动压缩并产出模型

自动压缩示例通过run.py脚本启动，会使用接口```paddleslim.auto_compression.AutoCompression```对模型进行自动压缩。配置config文件中训练部分的参数，将任务名称、模型类型、数据集名称、压缩参数传入，配置完成后便可对模型进行剪枝、蒸馏训练和离线量化。
数据集为CLUE，不同任务名称代表CLUE上不同的任务，可选择的任务名称有：afqmc, tnews, iflytek, ocnli, cmnli, cluewsc2020, csl。具体运行命令为
：
```shell
python run.py \
    --model_type='ppminilm' \
    --model_dir='./afqmc/' \
    --model_filename='inference.pdmodel' \
    --params_filename='inference.pdiparams' \
    --dataset='clue' \
    --save_dir='./save_afqmc_pruned/' \
    --batch_size=16 \
    --max_seq_length=128 \
    --task_name='afqmc' \
    --config_path='./configs/afqmc.yaml' 
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
  optim_args:
    weight_decay: 0.01
  optimizer: AdamW
  origin_metric: 0.7403
```

以下是默认的蒸馏、剪枝和离线量化的配置：

- 蒸馏参数

蒸馏参数包括teacher网络模型路径（即微调后未剪枝的模型），自动压缩策略会自动查找教师网络节点和对应的学生网络节点进行蒸馏，不需要手动设置。

```yaml
Distillation:
  teacher_model_dir: ./afqmc/
  teacher_model_filename: inference.pdmodel
  teacher_params_filename: inference.pdiparams
```

- 剪枝参数

剪枝参数包括裁剪算法和裁剪度。

```yaml
Prune:
  prune_algo: transformer_pruner
  pruned_ratio: 0.25
```

- 优化参数

```yaml
HyperParameterOptimization:
  batch_num:
  - 4
  - 16
  bias_correct:
  - true
  hist_percent:
  - 0.999
  - 0.99999
  max_quant_count: 20
  ptq_algo:
  - KL
  - hist
  weight_quantize_type:
  - channel_wise_abs_max
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

- [Paddle Inference Python部署](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/deployment/inference/python_inference.md)
- [Paddle Inference C++部署](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/deployment/inference/cpp_inference.md)
- [Paddle Lite部署](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/deployment/lite/lite.md)

## 6. FAQ