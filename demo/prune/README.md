# 图像分类模型卷积层通道剪裁示例

本示例将演示如何按指定的剪裁率对每个卷积层的通道数进行剪裁。该示例默认会自动下载并使用mnist数据。

当前示例支持以下分类模型：

- MobileNetV1
- MobileNetV2
- ResNet50
- PVANet


## 1. 数据准备

本示例支持`MNIST`和`ImageNet`两种数据。默认情况下，会自动下载并使用`MNIST`数据，如果需要使用`ImageNet`数据，请按以下步骤操作：

1). 根据分类模型中[ImageNet数据准备文档](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification#%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87)下载数据到`PaddleSlim/demo/data/ILSVRC2012`路径下。
2). 使用`train.py`脚本时，指定`--data`选项为`imagenet`.

## 2. 启动剪裁任务

通过以下命令启动裁剪任务：

```
export CUDA_VISIBLE_DEVICES=0
python train.py \
--model "MobileNet" \
--pruned_ratio 0.33 \
--data "imagenet"
```

其中，`model`用于指定待裁剪的模型。`pruned_ratio`用于指定各个卷积层通道数被裁剪的比例。`data`选项用于指定使用的数据集。

执行`python train.py --help`查看更多选项。

在本示例中，会在日志中输出剪裁前后的`FLOPs`，并且每训练一轮就会保存一个模型到文件系统。

## 3. 加载和评估模型

本节介绍如何加载训练过程中保存的模型。

执行以下代码加载模型并评估模型在测试集上的指标。

```
python eval.py \
--model "mobilenet" \
--data "mnist" \
--model_path "./models/0"
```

在脚本`eval.py`中，使用`paddleslim.prune.load_model`接口加载剪裁得到的模型。

## 4. 接口介绍

该示例使用了`paddleslim.Pruner`工具类，用户接口使用介绍请参考：[API文档](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/)

在调用`paddleslim.Pruner`工具类时，需要指定待裁卷积层的参数名称。不同模型的参数命名不同，
在`train.py`脚本中，提供了`get_pruned_params`方法，根据用户设置的选项`--model`确定要裁剪的参数。
