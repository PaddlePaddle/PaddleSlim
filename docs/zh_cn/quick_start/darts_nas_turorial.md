# 基于Darts搜索空间的SANAS实验教程

本教程说明了如何在Darts的基础上利用SANAS进行搜索实验并得到Darts_SA的结果，Darts_SA的Acc为97.27，相对Darts的模型（一下均成为Darts-model）结果来说精度提升0.141%， 模型大小下降11.2%。

## 搜索空间
本次实验仅会对Darts最终的模型结构的通道数进行进一步搜索，目的是得到一个精度更高并且模型参数更少的一个模型结构。所以定义如下搜索空间：
- 通道数`filter_num`: 定义了每个卷积操作的通道数变化区间。取值区间为：`[4, 8, 12, 16, 20, 36, 54, 72, 90, 108, 144, 180, 216, 252]`

按照通道数来定义Darts-model的block数量，则共有3个block，第一个block仅包含6个normal cell，之后的两个block每个block都包含和一个reduction cell和6个normal cell。每个cell中的所有卷积使用相同的通道数，所以共有20位token。

## 启动搜索

搜索文件位于: [darts_sanas_demo](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/nas/darts_nas.py)，搜索过程中限制模型参数量为不大于3.77M。
```python
cd demo/nas/
python darts_nas.py
```

## 启动最终实验
最终实验文件位于: [darts_sanas_demo](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/nas/darts_nas.py)，最终实验需要训练600epoch。以下示例输入token为`[5, 5, 0, 5, 5, 10, 7, 7, 5, 7, 7, 11, 10, 12, 10, 0, 5, 3, 10, 8]`。
```python
cd demo/nas/
python darts_nas.py --token 5 5 0 5 5 10 7 7 5 7 7 11 10 12 10 0 5 3 10 8 --retain_epoch 600
```
