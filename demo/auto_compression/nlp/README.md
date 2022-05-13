# 自然语言处理模型自动压缩示例

本示例将介绍如何使用PaddleNLP中Inference部署模型进行自动压缩。

## Benchmark
- PP-MiniLM模型

此实验首先对模型所有卷积裁剪25%，同时进行蒸馏训练，然后进行离线量化。

| 模型 | 策略 | AFQMC | TNEWS | IFLYTEK | CMNLI | OCNLI | CLUEWSC2020 | CSL | AVG |
|:------:|:------:|:------:|:------:|:------:|:------:|:-----------:|:------:|:------:|:------:|
| PP-MiniLM | Base模型| 74.03 | 56.66 | 60.21 | 80.98 | 76.20 | 84.21 | 77.36 | 72.81 |
| PP-MiniLM |剪枝蒸馏+离线量化| 73.56 | 56.38 | 59.87 | 80.80 | 76.44 | 82.23 | 77.77 | 72.44 |

性能测试的环境为
- 硬件：NVIDIA Tesla T4 单卡
- 软件：CUDA 11.0, cuDNN 8.0, TensorRT 8.0
- 测试配置：batch_size: 40, max_seq_len: 128

## 环境准备

### 1.准备数据
本案例默认以CLUE数据进行自动压缩实验，如数据集为非CLUE格式数据，请修改启动文本run.sh中dataset字段。

### 2.准备需要量化的环境
- PaddlePaddle >= 2.3
- PaddleNLP >= 2.3
注：安装PaddleNLP的目的是为了下载PaddleNLP中的数据集和Tokenizer。

### 3.准备待量化的部署模型
如果已经准备好部署的model.pdmodel和model.pdiparams部署模型，跳过此步。
根据[PaddleNLP文档](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples)导出Inference模型，本示例可参考[PaddleNLP PP-MiniLM 中文小模型](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/model_compression/pp-minilm)微调后保存下每个数据集下有最高准确率的模型。

## 开始自动压缩

### 进行剪枝蒸馏和离线量化自动压缩

蒸馏量化自动压缩示例通过run.py脚本启动，会使用接口paddleslim.auto_compression.AutoCompression对模型进行量化训练。将任务名称、模型类型、数据集名称、压缩参数传入，对模型进行剪枝、蒸馏训练和离线量化。数据集为CLUE，不同任务名称代表不同CLUE上的任务，可选择的任务名称有：afqmc, tnews, iflytek, ocnli, cmnli, cluewsc2020, csl。具体运行命令为：
```shell
python run.py \
    --model_type='ppminilm' \
    --model_dir='./afqmc_base/' \
    --model_filename='inference.pdmodel' \
    --params_filename='inference.pdiparams' \
    --dataset='clue' \
    --save_dir='./save_afqmc_pruned/' \
    --batch_size=16 \
    --max_seq_length=128 \
    --task_name='afqmc' \
    --config_path='./configs/afqmc.yaml' 
```