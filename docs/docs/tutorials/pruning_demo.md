# 卷积通道剪裁示例

本示例将演示如何按指定的剪裁率对每个卷积层的通道数进行剪裁。该示例默认会自动下载并使用mnist数据。

当前示例支持以下分类模型：

- MobileNetV1
- MobileNetV2
- ResNet50
- PVANet

## 接口介绍

该示例使用了`paddleslim.Pruner`工具类，用户接口使用介绍请参考：[API文档](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/)

## 确定待裁参数

不同模型的参数命名不同，在剪裁前需要确定待裁卷积层的参数名称。可通过以下方法列出所有参数名：

```python
for param in program.global_block().all_parameters():
    print("param name: {}; shape: {}".format(param.name, param.shape))
```

在`train.py`脚本中，提供了`get_pruned_params`方法，根据用户设置的选项`--model`确定要裁剪的参数。

## 启动裁剪任务

通过以下命令启动裁剪任务：

```python
export CUDA_VISIBLE_DEVICES=0
python train.py
```

执行`python train.py --help`查看更多选项。

## 注意

1. 在接口`paddle.Pruner.prune`的参数中，`params`和`ratios`的长度需要一样。
