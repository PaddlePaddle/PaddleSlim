# HuggingFace 预训练模型压缩部署示例
目录：
- [1. 简介](#1简介)
- [2. Benchmark](#2Benchmark)
- [3. 自动压缩流程](#自动压缩流程)
  - [3.1 准备环境](#31-准备环境)
  - [3.2 准备数据集](#32-准备数据集)
  - [3.3 X2Paddle转换模型流程](#33-X2Paddle转换模型流程)
  - [3.4 自动压缩并产出模型](#34-自动压缩并产出模型)
- [4. 压缩配置介绍](#4压缩配置介绍)
- [5. 预测部署](#5预测部署)
- [6. FAQ](6FAQ)

## 1. 简介
飞桨模型转换工具[X2Paddle](https://github.com/PaddlePaddle/X2Paddle)支持将```Caffe/TensorFlow/ONNX/PyTorch```的模型一键转为飞桨（PaddlePaddle）的预测模型。借助X2Paddle的能力，PaddleSlim的自动压缩功能可方便地用于各种框架的推理模型。

本示例将以[Pytorch](https://github.com/pytorch/pytorch)框架的自然语言处理模型为例，介绍如何自动压缩其他框架中的自然语言处理模型。本示例会利用[huggingface](https://github.com/huggingface/transformers)开源transformers库，将Pytorch框架模型转换为Paddle框架模型，再使用ACT自动压缩功能进行自动压缩。本示例使用的自动压缩策略为剪枝蒸馏和量化训练。


## 2. Benchmark
[BERT](https://arxiv.org/abs/1810.04805) （```Bidirectional Encoder Representations from Transformers```）以Transformer 编码器为网络基本组件，使用掩码语言模型（```Masked Language Model```）和邻接句子预测（```Next Sentence Prediction```）两个任务在大规模无标注文本语料上进行预训练（pre-train），得到融合了双向内容的通用语义表示模型。以预训练产生的通用语义表示模型为基础，结合任务适配的简单输出层，微调（fine-tune）后即可应用到下游的NLP任务，效果通常也较直接在下游的任务上训练的模型更优。此前BERT即在[GLUE](https://gluebenchmark.com/tasks)评测任务上取得了SOTA的结果。

基于bert-base-cased模型，压缩前后的精度如下：
| 模型 | 策略 | CoLA | MRPC | QNLI | QQP | RTE | SST2  | STSB  | AVG |
|:------:|:------:|:------:|:------:|:-----------:|:------:|:------:|:------:|:------:|:------:|
| bert-base-cased | Base模型 | 60.06 | 84.31 | 90.68 | 90.84 | 63.53 | 91.63  | 88.46 |  81.35  |
| bert-base-cased | 剪枝蒸馏+量化训练 | 58.69 | 85.05 | 90.74 | 90.42 | 65.34 | 92.08 | 88.22 |  81.51 |


|  模型 |策略| Accuracy（avg） | 预测时延<sup><small>FP32</small><sup><br><sup> | 预测时延<sup><small>FP16</small><sup><br><sup> | 预测时延<sup><small>INT8</small><sup><br><sup> | 加速比 |
|:-------:|:----------:|:------------:|:------:|:------:|:------:|:------:|
| bert-base-uncased | Base模型 |  92.66 | 173.00ms | 38.42ms | - | - |
| bert-base-uncased | 剪枝+量化训练 | 92.31 | - | - | 33.24ms | 5.20x |

- Nvidia GPU 测试环境：
  - 硬件：NVIDIA Tesla T4 单卡
  - 软件：CUDA 11.2, cuDNN 8.1, TensorRT 8.6.1.6
  - 测试配置：batch_size: 32, seqence length: 128

## 3. 自动压缩流程
#### 3.1 准备环境
- python >= 3.6
- PaddlePaddle ==2.6 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim ==2.6
- X2Paddle develop版本
- transformers >= 4.18.0
- PaddleNLP 2.7.2
- tensorflow == 1.14 (如需压缩TensorFlow模型)
- onnx  1.15.0 (如需压缩ONNX模型)
- torch 1.13.1 (如需压缩PyTorch模型)

安装paddlepaddle：
```shell
# CPU
python -m pip install paddlepaddle==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
# GPU 以CUDA11.2为例
python -m pip install paddlepaddle-gpu==2.6.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

安装paddleslim：
```shell
pip install paddleslim
```

安装X2Paddle:
```
git clone https://github.com/PaddlePaddle/X2Paddle.git
cd X2Paddle
git checkout develop
python setup.py install
```

安装paddlenlp：
```shell
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

注：安装PaddleNLP的目的是为了下载PaddleNLP中的数据集。


#### 3.2 准备数据集
本案例默认以GLUE数据进行自动压缩实验，PaddleNLP会自动下载对应数据集。


#### 3.3 X2Paddle转换模型流程

**方式1: PyTorch2Paddle直接将Pytorch动态图模型转为Paddle静态图模型**

```shell
import torch
import numpy as np
# 将PyTorch模型设置为eval模式
torch_model.eval()
# 构建输入，
input_ids = torch.zeros([batch_size, max_length]).long()
token_type_ids = torch.zeros([batch_size, max_length]).long()
attention_msk = torch.zeros([batch_size, max_length]).long()
# 进行转换
from x2paddle.convert import pytorch2paddle
pytorch2paddle(torch_model,
               save_dir='./x2paddle_cola/',
               jit_type="trace",
               input_examples=[input_ids, attention_msk, token_type_ids])
```

PyTorch2Paddle支持trace和script两种方式的转换，均是PyTorch动态图到Paddle动态图的转换，转换后的Paddle动态图运用动转静可转换为静态图模型。
- jit_type为"trace"时，input_examples不可为None，转换后自动进行动转静，输入shape固定。
- jit_type为"script"时，当input_examples为None时，只生成动态图代码；当input_examples不为None时，才能自动动转静。

注意：
- 由于自动压缩的是静态图模型，所以这里需要将```jit_type```设置为```trace```，并且注意PyTorch模型中需要设置```pad_to_max_length```，且设置的```max_length```需要和转换时构建的数据相同。
- HuggingFace默认输入```attention_mask```，PaddleNLP默认不输入，这里需要保持一致。可以PaddleNLP中设置```return_attention_mask=True```。
- 使用PaddleNLP的tokenizer时需要在模型保存的文件夹中加入tokenizer的配置文件，可使用PaddleNLP中训练后自动保存的 ```model_config.json，special_tokens_map.json, tokenizer_config.json, vocab.txt```，也可使用Huggingface训练后自动保存的 ```config.json，special_tokens_map.json, tokenizer_config.json, vocab.txt```。


更多Pytorch2Paddle示例可参考[PyTorch模型转换文档](https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/inference_model_convertor/pytorch2paddle.md)。其他框架转换可参考[X2Paddle模型转换工具](https://github.com/PaddlePaddle/X2Paddle)

如想快速尝试运行实验，也可以直接下载已经转换好的模型，链接如下：
| [CoLA](https://paddle-slim-models.bj.bcebos.com/act/x2paddle_cola.tar) | [MRPC](https://paddle-slim-models.bj.bcebos.com/act/x2paddle_mrpc.tar) | [QNLI](https://paddle-slim-models.bj.bcebos.com/act/x2paddle_qnli.tar) | [QQP](https://paddle-slim-models.bj.bcebos.com/act/x2paddle_qqp.tar) | [RTE](https://paddle-slim-models.bj.bcebos.com/act/x2paddle_rte.tar) | [SST2](https://paddle-slim-models.bj.bcebos.com/act/x2paddle_sst2.tar) | [STSB](https://paddle-slim-models.bj.bcebos.com/act/x2paddle_stsb.tar) |

```shell
wget https://paddle-slim-models.bj.bcebos.com/act/x2paddle_cola.tar
tar xf x2paddle_cola.tar
```

**方式2: Onnx2Paddle将Pytorch动态图模型保存为Onnx格式后再转为Paddle静态图模型**


PyTorch 导出 ONNX 动态图模型
```shell
torch_model.eval()
input_ids = torch.unsqueeze(torch.tensor([0] * args.max_length), 0)
token_type_ids = torch.unsqueeze(torch.tensor([0] * args.max_length), 0)
attention_mask = torch.unsqueeze(torch.tensor([0] * args.max_length), 0)
input_names = ['input_ids', 'attention_mask', 'token_type_ids']
output_names = ['output']
torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        'model.onnx',
        opset_version=11,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={'input_ids': [0], 'attention_mask': [0], 'token_type_ids': [0]})
```

通过 X2Paddle 命令导出 Paddle 模型
```shell
x2paddle --framework=onnx --model=model.onnx --save_dir=pd_model_dynamic
```

在自动生成的 x2paddle_code.py 中添加如下代码：
```shell
def main(x0, x1, x2):
    # x0, x1, x2 为模型输入.
    paddle.disable_static()
    params = paddle.load('model.pdparams')
    model = BertForSequenceClassification()
    model.set_dict(params)
    model.eval()
    ## convert to jit
    sepc_list = list()
    sepc_list.append(
            paddle.static.InputSpec(
                shape=[-1, 128], name="x2paddle_input_ids", dtype="int64"),
            paddle.static.InputSpec(
                shape=[-1, 128], name="x2paddle_attention_mask", dtype="int64"),
            paddle.static.InputSpec(
                shape=[-1, 128], name="x2paddle_token_type_ids", dtype="int64"))
    static_model = paddle.jit.to_static(model, input_spec=sepc_list)
    paddle.jit.save(static_model, "./x2paddle_cola")
```


#### 3.4 自动压缩并产出模型
以```cola```任务为例，在配置文件```./config/cola.yaml```中配置推理模型路径、压缩策略参数等信息，并通过```--config_path```将配置文件传给示例脚本```run.py```。
在```run.py```中，调用接口```paddleslim.auto_compression.AutoCompression```加载配置文件，使用以下命令对推理模型进行自动压缩：

```shell
export CUDA_VISIBLE_DEVICES=0
python run.py --config_path=./configs/cola.yaml --save_dir='./output/cola/'
```

如仅需验证模型精度，或验证压缩之后模型精度，在启动```run.py```脚本时，将配置文件中模型文件夹 ```model_dir``` 改为压缩之后保存的文件夹路径 ```./output/cola``` ，命令加上```--eval True```即可：
```shell
export CUDA_VISIBLE_DEVICES=0
python run.py --config_path=./configs/cola.yaml  --eval True
```

## 4. 预测部署

量化模型在GPU上可以使用TensorRT进行加速，在CPU上可以使用MKLDNN进行加速。

以下字段用于配置预测参数：

| 参数名 | 含义 |
|:------:|:------:|
| model_path | inference 模型文件所在目录,该目录下需要有文件 model.pdmodel 和 model.pdiparams 两个文件 |
| model_filename | 模型文件的名称,默认值为model.pdmodel |
| params_filename | 参数文件的名称,默认值为model.pdiparams |
| task_name | 要执行的任务名称,默认为cola,这里指定的任务应该是"METRIC_CLASSES"字典中包含的任务之一 |
| model_type | 选择的模型类型,默认为bert-base-cased。这里指定了预训练模型的类型或架构 |
| model_name_or_path | 模型的目录或名称,默认为bert-based-cased。这里可以指定一个预训练模型的目录或HuggingFace预训练模型的模型名称 |
| device | 选择用于推理的设备，默认为gpu,这里可以是gpu或cpu |
| batch_size | 预测的批处理大大小，默认为32 |
| max_seq_length | 输入序列的最大长度，默认为128,超过这个长度的序列将被截断，短于这个长度的序列将被填充 |
| perf_warmup_steps | 性能测试的预热步骤数,默认为20,这是在正式计算推理性能前,进行的预热迭代次数,以确保性能稳定 |
| use_trt | 是否使用TensorRT进行推理 |
| precision | 推理精度，默认为fp32,可设置为fp16或int8 |
| use_mkldnn | 是否使用MKLDNN进行推理,默认为False。这是针对CPU推理时,是否启用MKL-DNN进行加速 |
| cpu_threads | CPU线程数,默认为10。这是针对CPU推理时,指定使用的线程数 |

- Paddle-TensorRT预测：

环境配置：如果使用 TesorRT 预测引擎，需安装 ```WITH_TRT=ON``` 的Paddle，下载地址：[Python预测库](https://paddleinference.paddlepaddle.org.cn/master/user_guides/download_lib.html#python)

首先下载量化好的模型：
```shell
wget https://bj.bcebos.com/v1/paddle-slim-models/act/x2paddle_cola_new_calib.tar
tar -xf x2paddle_cola_new_calib.tar
```

```shell
python paddle_inference_eval.py \
      --model_path=x2paddle_cola_new_calib \
      --use_trt \
      --precision=int8 \
      --batch_size=1
```

- MKLDNN预测：

```shell
python paddle_inference_eval.py \
      --model_path=x2paddle_cola_new_calib \
      --device=cpu \
      --use_mkldnn=True \
      --cpu_threads=10 \
      --batch_size=1 \
      --precision=int8
```
bert-base-uncased模型
```shell
python paddle_inference_eval.py \
      --model_path=infer_model \
      --use_trt \
      --precision=fp32 \
      --batch_size=1
```
```shell
python paddle_inference_eval.py  \
       --model_path=output/unsst2 \
       --use_trt \
       --precision=int8 \
       --batch_size=32 \
       --task_name=sst-2
```



## 5. FAQ
