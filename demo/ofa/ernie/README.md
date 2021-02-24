# OFA压缩ERNIE模型

ERNIE是百度开创性提出的基于知识增强的持续学习语义理解框架，该框架将大数据预训练与多源丰富知识相结合，通过持续学习技术，不断吸收海量文本数据中词汇、结构、语义等方面的知识，实现模型效果不断进化。本教程讲介绍如何使用PaddleSlim对[ERNIE](https://github.com/PaddlePaddle/ERNIE)模型进行压缩。

本教程只会演示如何快速启动相应训练，详细教程请参考：[ERNIE](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/nlp/ernie_slim_ofa_tutorial.md)

使用本教程压缩算法可以在精度无损的情况下，对原始Tiny-ERNIE模型进行40%的加速。

## 1. 快速开始
本教程以 CLUE/XNLI 数据集为例。

### 1.1 安装依赖项
由于ERNIE repo中动态图模型是基于Paddle 1.8.5版本进行开发的，所以本教程依赖Paddle 1.8.5和Paddle-ERNIE 0.0.4.dev1.

```shell
pip install paddle-ernie==0.0.4.dev1
pip install paddlepaddle_gpu==1.8.5.post97
```

propeller是ERNIE框架中辅助模型训练的高级框架，包含NLP常用的前、后处理流程。你可以通过将ERNIE repo根目录放入PYTHONPATH的方式导入propeller:
```shell
git clone https://github.com/PaddlePaddle/ERNIE
cd ERNIE
export PYTHONPATH=$PWD:$PYTHONPATH
```

### 1.2 Fine-tuning
首先需要对Pretrain-Model在实际的下游任务上进行Fine-tuning，得到需要压缩的模型。参考[Fine-tuning教程](https://github.com/PaddlePaddle/ERNIE/tree/v2.4.0#%E6%94%AF%E6%8C%81%E7%9A%84nlp%E4%BB%BB%E5%8A%A1)得到Tiny-ERNIE模型在XNLI数据集上的Fine-tuning模型.

### 1.3 压缩训练

```python
python ./ofa_ernie.py \
       --from_pretrained ernie-tiny \
       --data_dir ./data/xnli \
       --width_mult_list 1.0 0.75 0.5 0.25 \
       --depth_mult_list 1.0 0.75
```
其中参数释义如下：
- `from_pretrained` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。若模型相关内容保存在本地，这里也可以提供相应目录地址。
- `data_dir` 指明数据保存目录。
- `width_mult_list` 表示压缩训练过程中，对每层Transformer Block的宽度选择的范围。
- `depth_mult_list` 表示压缩训练过程中，模型包含的Transformer Block数量的选择的范围。

## 2. OFA接口介绍
OFA API介绍参考[API](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/dygraph/ofa/ofa_api.rst)
