# OFA压缩ERNIE模型

ERNIE是百度开创性提出的基于知识增强的持续学习语义理解框架，该框架将大数据预训练与多源丰富知识相结合，通过持续学习技术，不断吸收海量文本数据中词汇、结构、语义等方面的知识，实现模型效果不断进化。本教程讲介绍如何使用PaddleSlim对[ERNIE](https://github.com/PaddlePaddle/ERNIE)模型进行压缩。

使用本教程压缩算法可以在精度无损的情况下，对原始Tiny-ERNIE模型进行40%的加速。

## 快速开始
本教程以 CLUE/XNLI 数据集为例。

### 安装依赖项
由于ERNIE repo中动态图模型是基于Paddle 1.8版本进行开发的，所以本教程依赖Paddle 1.8.5和Paddle-ERNIE.

```shell
pip install paddle-ernie
pip install paddlepaddle_gpu==1.8.5.post97
```

propeller是ERNIE框架中辅助模型训练的高级框架，包含NLP常用的前、后处理流程。你可以通过将本repo根目录放入PYTHONPATH的方式导入propeller:
```shell
git clone https://github.com/PaddlePaddle/ERNIE
cd ERNIE
export PYTHONPATH=$PWD:$PYTHONPATH
```

### Fine-tuning
首先需要对Pretrain-Model在实际的下游任务上进行Fine-tuning，得到需要压缩的模型。参考[Fine-tuning教程](https://github.com/PaddlePaddle/ERNIE/tree/v2.4.0#%E6%94%AF%E6%8C%81%E7%9A%84nlp%E4%BB%BB%E5%8A%A1)得到Tiny-ERNIE模型在XNLI数据集上的Fine-tuning模型.

### 压缩训练

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

## OFA接口介绍
TODO

## 原理介绍

1. 对Fine-tuning得到模型通过计算参数及其梯度的乘积得到参数的重要性，把模型参数根据重要性进行重排序。
2. 超网络中最大的子网络选择和Bert-base模型网络结构一致的网络结构，其他小的子网络是对最大网络的进行不同的宽度选择来得到的，宽度选择
具体指的是网络中的参数进行裁剪，所有子网络在整个训练过程中都是参数共享的。
2. 用重排序之后的模型参数作为超网络模型的初始化参数。
3. Fine-tuning之后的模型作为教师网络，超网络作为学生网络，进行知识蒸馏。

<p align="center">
<img src="../../../docs/images/algo/ofa_bert.jpg" width="950"/><br />
整体流程图
</p>
