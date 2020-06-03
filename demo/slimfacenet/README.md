# slimfacenet使用示例

本示例将演示如何训练`slimfacenet`及评测`slimfacenet`量化模型。

本示例依赖：Paddle 1.8 PaddleSlim 1.1.1

当前示例支持以下人脸识别模型：

- `SlimFaceNet_A_x0_60`
- `SlimFaceNet_B_x0_75`

## 1. 数据准备

本示例支持`CASIA`和`lfw`两种公开数据集默认情况：

1). 训练数据集位置`./CASIA`
2). 测试数据集位置`./lfw`

## 2. 下载预训练模型

如果使用预先训练并量化好的`slimfacenet`模型，可以从以下地址下载


## 3. 启动`slimfacenet`训练任务

通过以下命令启动训练任务：

```
sh slim_train.sh
或者
export CUDA_VISIBLE_DEVICES=0
python -u train_eval.py \
    --action train \
    --model=SlimFaceNet_B_x0_75
```

其中，SlimFaceNet_A_x0_60是`slimfacenet`搜索空间中的一个模型结构，通道数的缩放系数为0.6，
在每个缩放系数下搜索空间中都共有6**15(约4700亿)种不同的模型结构。模型训练好之后会保存在`./out_inference/`


## 4. 将float32模型量化为int8模型

通过以下命令启动训练任务：

```
sh slim_quant.sh
或者
export CUDA_VISIBLE_DEVICES=0
python -u train_eval.py --action quant
```
执行完之后量化模型会保存在`./quant_model/`, 注当前阶段量化模型还是是按float32保存的，转paddlelite后会变为int8

## 4. 加载和评估量化模型

本节介绍如何加载并评测预先训练好并量化后的模型。

执行以下代码加载模型并评估模型在测试集上的指标。

```
将量化模型默认地址在`./quant_model/`
sh slim_eval.sh
或者
export CUDA_VISIBLE_DEVICES=0
python train_eval.py --action test
```
