# slimfacenet使用示例

本示例将演示如何训练`slimfacenet`及评测`slimfacenet`量化模型。

当前示例支持以下人脸识别模型：

- `slimfacenet`


## 1. 数据准备

本示例支持`CASIA`和`lfw`两种公开数据集默认情况：

1). 训练数据集位置`./CASIA`
2). 测试数据集位置`./lfw`

## 2. 下载预训练模型

如果使用预先训练并量化好的`slimfacenet`模型，可以从以下地址下载


## 3. 启动`slimfacenet`训练任务

通过以下命令启动训练任务：

```
sh slim_train.sh 0.75 1,0,1,3,0,3,1,1,1,1,0,0,2,5,3
或者
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH='PATH to CUDA and CUDNN'
python -u train_eval.py \
    --arch=1,0,1,3,0,3,1,1,1,1,0,0,2,5,3 \
    --action final \
    --scale=0.75
```

其中，`arch`=1,0,1,3,0,3,1,1,1,1,0,0,2,5,3为`slimfacenet`搜索空间中的一个模型结构，`scale`=0.75为通道数的缩放系数，
在每个`scale`下搜索空间中都共有6**15(约4700亿)种不同的模型结构。


## 4. 加载和评估量化模型

本节介绍如何加载并评测预先训练好并量化后的模型。

执行以下代码加载模型并评估模型在测试集上的指标。

```
将量化模型放在./quant_model/
sh slim_eval.sh
或者
export CUDA_VISIBLE_DEVICES=0
python train_eval.py --action test
```
