代码路径：https://github.com/wanghaoshuang/PaddleSlim/tree/develop/demo/sensitive

该示例介绍如何分析卷积网络中各卷积层的敏感度，以及如何根据计算出的敏感度选择一组合适的剪裁率。
该示例默认会自动下载并使用mnist数据。支持以下模型：

- MobileNetV1
- MobileNetV2
- ResNet50

## 1. 接口介绍

该示例涉及以下接口：

- [paddleslim.prune.sensitivity](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/#sensitivity)
- [paddleslim.prune.merge_sensitive](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/#merge_sensitive)
- [paddleslim.prune.get_ratios_by_loss](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/#get_ratios_by_losssensitivities-loss)

## 2. 运行示例

执行以下代码运行示例：

```
export CUDA_VISIBLE_DEVICES=0
python train.py --model "MobileNetV1"
```

通过`python train.py --help`查看更多选项。

## 3. 重要步骤说明

### 3.1 计算敏感度

计算敏感度之前，用户需要搭建好用于测试的网络，以及实现评估模型精度的回调函数。

调用`paddleslim.prune.sensitivity`接口计算敏感度。敏感度信息会追加到`sensitivities_file`选项所指定的文件中，如果需要重新计算敏感度，需要先删除`sensitivities_file`文件。

如果模型评估速度较慢，可以通过多进程的方式加速敏感度计算过程。比如在进程1中设置`pruned_ratios=[0.1, 0.2, 0.3, 0.4]`，并将敏感度信息存放在文件`sensitivities_0.data`中，然后在进程2中设置`pruned_ratios=[0.5, 0.6, 0.7]`，并将敏感度信息存储在文件`sensitivities_1.data`中。这样每个进程只会计算指定剪切率下的敏感度信息。多进程可以运行在单机多卡，或多机多卡。

代码如下：

```
# 进程1
sensitivity(
    val_program,
    place,
    params,
    test,
    sensitivities_file="sensitivities_0.data",
    pruned_ratios=[0.1, 0.2, 0.3, 0.4])
```

```
# 进程2
sensitivity(
    val_program,
    place,
    params,
    test,
    sensitivities_file="sensitivities_1.data",
    pruned_ratios=[0.5, 0.6, 0.7])
```


### 3.2 合并敏感度

如果用户通过上一节多进程的方式生成了多个存储敏感度信息的文件，可以通过`paddleslim.prune.merge_sensitive`将其合并，合并后的敏感度信息存储在一个`dict`中。代码如下：

```
sens = merge_sensitive(["./sensitivities_0.data", "./sensitivities_1.data"])
```

### 3.3 计算剪裁率

调用`paddleslim.prune.get_ratios_by_loss`接口计算一组剪裁率。

```
ratios = get_ratios_by_loss(sens, 0.01)
```

其中，`0.01`为一个阈值，对于任意卷积层，其剪裁率为使精度损失低于阈值`0.01`的最大剪裁率。

用户在计算出一组剪裁率之后可以通过接口`paddleslim.prune.Pruner`剪裁网络，并用接口`paddleslim.analysis.flops`计算`FLOPs`。如果`FLOPs`不满足要求，调整阈值重新计算出一组剪裁率。
