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
本示例将以自然语言处理模型PP-MiniLM和ERNIE 3.0-Medium为例，介绍如何使用PaddleNLP中Inference部署模型进行自动压缩.

## 2. Benchmark

- PP-MiniLM: 6层的预训练中文小模型，使用PaddleNLP中```from_pretrained```导入PP-MiniLM之后，就可以在自己的数据集上进行fine-tuning，具体介绍可参考[PP-MiniLM文档](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/model_compression/pp-minilm#PP-MiniLM%E4%B8%AD%E6%96%87%E5%B0%8F%E6%A8%A1%E5%9E%8B)。
- ERNIE 3.0-Medium:  中文预训练模型, 关键参数为(6-layer, 768-hidden, 12-heads), 详情请参考[PaddleNLP ERNIE 3.0](https://github.com/PaddlePaddle/PaddleNLP/tree/v2.3.3/model_zoo/ernie-3.0)


模型精度对比如下：
| 模型 | 策略 | AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUEWSC2020 | CSL | AVG |
|:------:|:------:|:------:|:------:|:------:|:------:|:-----------:|:------:|:------:|:------:|
| PP-MiniLM | Base模型| 74.03 | 56.66 | 60.21 | 80.98 | 76.20 | 84.21 | 77.36 | 72.81 |
| PP-MiniLM |剪枝蒸馏+离线量化| 74.03 | 56.62 | 60.18 | 80.87 | 75.28 | 80.92 | 75.03 | 71.85 |
| ERNIE 3.0-Medium | Base模型| 75.35 | 57.45 | 60.17 | 81.16 | 77.19 | 80.59 | 79.70 | 73.09 |
| ERNIE 3.0-Medium | 剪枝+量化训练| 74.17 | 56.84 | 59.75 | 80.54 | 76.03 | 76.97 | 80.80 | 72.16 |


模型在不同任务上平均精度以及加速对比如下：
|  模型 |策略| Accuracy（avg） | 预测时延<sup><small>FP32</small><sup><br><sup> | 预测时延<sup><small>FP16</small><sup><br><sup> | 预测时延<sup><small>INT8</small><sup><br><sup> | 加速比 |
|:-------:|:--------:|:----------:|:------------:|:------:|:------:|:------:|
|PP-MiniLM| Base模型|  72.81 | 94.49ms | 23.31ms | - |  - |
|PP-MiniLM| 剪枝+离线量化 |  71.85 | - | - | 15.76ms | 5.99x |
|ERNIE 3.0-Medium| Base模型| 73.09  | 89.71ms | 20.76ms | - | - |
|ERNIE 3.0-Medium| 剪枝+量化训练 |  72.16 | - | - | 14.08ms | 6.37x |

性能测试的环境为
- 硬件：NVIDIA Tesla T4 单卡
- 软件：CUDA 11.2, cuDNN 8.1, TensorRT 8.4
- 测试配置：batch_size: 32, max_seq_len: 128

## 3. 自动压缩流程

#### 3.1 准备环境
- python >= 3.6
- PaddlePaddle ==2.5 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim ==2.5
- PaddleNLP ==2.6

安装paddlepaddle：
```shell
# CPU
pip install paddlepaddle==2.5.0
# GPU 以Ubuntu、CUDA 11.2为例
python -m pip install paddlepaddle-gpu==2.5.0.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
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

| 模型 | AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUEWSC2020 | CSL |
|:------:|:------:|:------:|:------:|:------:|:-----------:|:------:|:------:|
| PP-MiniLM | [afqmc](https://bj.bcebos.com/v1/paddle-slim-models/act/afqmc.tar) | [tnews](https://bj.bcebos.com/v1/paddle-slim-models/act/tnews.tar) | [iflytek](https://bj.bcebos.com/v1/paddle-slim-models/act/iflytek.tar) | [cmnli](https://bj.bcebos.com/v1/paddle-slim-models/act/cmnli.tar) | [ ocnli](https://bj.bcebos.com/v1/paddle-slim-models/act/ocnli.tar) | [cluewsc2020](https://bj.bcebos.com/v1/paddle-slim-models/act/cluewsc.tar) | [csl](https://bj.bcebos.com/v1/paddle-slim-models/act/csl.tar) |
| ERNIE 3.0-Medium | [afqmc](https://bj.bcebos.com/v1/paddle-slim-models/act/NLP/ernie3.0-medium/fp32_models/AFQMC.tar) | [tnews](https://bj.bcebos.com/v1/paddle-slim-models/act/NLP/ernie3.0-medium/fp32_models/TNEWS.tar) | [iflytek](https://bj.bcebos.com/v1/paddle-slim-models/act/NLP/ernie3.0-medium/fp32_models/IFLYTEK.tar) | [cmnli](https://bj.bcebos.com/v1/paddle-slim-models/act/NLP/ernie3.0-medium/fp32_models/CMNLI.tar) | [ocnli](https://bj.bcebos.com/v1/paddle-slim-models/act/NLP/ernie3.0-medium/fp32_models/OCNLI.tar) | [cluewsc2020](https://bj.bcebos.com/v1/paddle-slim-models/act/NLP/ernie3.0-medium/fp32_models/CLUEWSC2020.tar) | [csl](https://bj.bcebos.com/v1/paddle-slim-models/act/NLP/ernie3.0-medium/fp32_models/CSL.tar) |

从上表获得模型超链接, 并用以下命令下载推理模型文件:

```shell
wget https://bj.bcebos.com/v1/paddle-slim-models/act/afqmc.tar
tar -zxvf afqmc.tar
```

##### 重新微调模型

可参考[PaddleNLP PP-MiniLM 中文小模型](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/model_compression/pp-minilm)微调后保存下每个数据集下有最高准确率的模型。
其他模型可根据[PaddleNLP文档](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples)导出Inference模型。

#### 3.4 自动压缩并产出模型

自动压缩示例通过run.py脚本启动，会使用接口```paddleslim.auto_compression.AutoCompression```对模型进行自动压缩。配置config文件中训练部分的参数，将任务名称、模型类型、数据集名称、压缩参数传入，配置完成后便可对模型进行剪枝、蒸馏训练和离线量化。
数据集为CLUE，不同任务名称代表CLUE上不同的任务，可选择的任务名称有：```afqmc, tnews, iflytek, ocnli, cmnli, cluewsc2020, csl```。具体运行命令为：
：
```shell
export CUDA_VISIBLE_DEVICES=0
python run.py --config_path='./configs/pp-minilm/auto/afqmc.yaml' --save_dir='./save_afqmc_pruned/'
```


如仅需验证模型精度，或验证压缩之后模型精度，在启动```run.py```脚本时，将配置文件中模型文件夹 ```model_dir``` 改为压缩之后保存的文件夹路径 ```./save_afqmc_pruned``` ，命令加上```--eval True```即可：
```shell
export CUDA_VISIBLE_DEVICES=0
python run.py --config_path='./configs/pp-minilm/auto/afqmc.yaml'  --eval True
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
  teacher_model_dir: ./afqmc
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

- 离线量化超参搜索

本示例的离线量化采取了超参搜索策略，以选择最优的超参数取得更好的离线量化效果。首先，配置待搜索的参数：

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

其次，配置离线量化参数：

量化参数主要设置量化比特数和量化op类型，其中量化op包含卷积层（conv2d, depthwise_conv2d）和全连接层（mul，matmul_v2）。

```yaml
QuantPost:
  activation_bits: 8
  quantize_op_types:
  - conv2d
  - depthwise_conv2d
  - mul
  - matmul_v2
  weight_bits: 8
```

## 5. 预测部署


量化模型在GPU上可以使用TensorRT进行加速，在CPU上可以使用MKLDNN进行加速。

以下字段用于配置预测参数：

| 参数名 | 含义 |
|:------:|:------:|
| model_path | inference 模型文件所在目录，该目录下需要有文件 model.pdmodel 和 model.pdiparams 两个文件 |
| model_filename | 模型文件的名称，默认值为inference.pdmodel |
| params_filename | 参数文件的名称，默认值为inference.pdiparams |
| task_name | 要执行的任务名称，默认值为afqmc |
| dataset | 模型使用的数据集，默认值为clue |
| device | 用于推理的设备，默认为gpu，可选cpu或gpu |
| batch_size | 推理时的batch size，默认为32 |
| max_seq_len | 输入序列在分词后的最大长度,默认值为128,如果序列长于此值，将会被截断;如果短于此值，将会被填充|
| perf_warmup_steps | 性能测试的预热步数,默认值为20 |
| use_trt | 一个标志(flag),用于决定是否使用TensorRT推理 |
| precision | 推理精度，默认为fp32,可选fp16或int8 |
| use_mkldnn | 一个标志(flag),用于决定是否使用MKLDNN推理 |
| cpu_threads | CPU线程数，默认为1 |


- TensorRT预测：

首先下载量化好的模型：
```shell
wget https://bj.bcebos.com/v1/paddle-slim-models/act/save_ppminilm_afqmc_new_calib.tar
tar -xf save_ppminilm_afqmc_new_calib.tar
```

```shell
python paddle_inference_eval.py \
      --model_path=save_ppminilm_afqmc_new_calib \
      --model_filename=inference.pdmodel \
      --params_filename=inference.pdiparams \
      --task_name='afqmc' \
      --use_trt \
      --precision=int8
```

- ERNIE 3.0-Medium:
```shell
python paddle_inference_eval.py \
      --model_path=TNEWS \
      --model_filename=infer.pdmodel \
      --params_filename=infer.pdiparams \
      --task_name='tnews' \
      --use_trt \
      --precision=fp32
```
```shell
python paddle_inference_eval.py \
      --model_path=save_tnews_pruned \
      --model_filename=infer.pdmodel \
      --params_filename=infer.pdiparams \
      --task_name='tnews' \
      --use_trt \
      --precision=int8
```

- MKLDNN预测：

```shell
python paddle_inference_eval.py \
      --model_path=save_ppminilm_afqmc_new_calib \
      --model_filename=inference.pdmodel \
      --params_filename=inference.pdiparams \
      --task_name='afqmc' \
      --device=cpu \
      --use_mkldnn=True \
      --cpu_threads=10 \
      --precision=int8
```

## 6. FAQ
