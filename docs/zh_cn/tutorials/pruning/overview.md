# Overview

PaddleSlim提供以下内置剪裁方法。


| 序号 | 剪裁方法           | 支持静态图 | 支持动态图 | 支持敏感度分析 | 支持自定义各层剪裁率|
|:----:|:------------------:|:----------:|:----------:|:--------------:|:-------------------:|
|1     |FPGMFilterPruner    |是          | 是         | 是             |是                   |
|2     |L1NormFilterPruner  |是          | 是         | 是             |是                   |
|3     |L2NormFilterPruner  |否          | 是         | 是             |是                   |
|4     |SlimFilterPruner    |是          | 否         | 是             |是                   |
|5     |OptSlimFilterPruner |是          | 否         | 是             |是                   |

注：

- 支持敏感度分析：意为是否支持通过各个层的敏感度分析来确定各个卷积层的剪裁率。

- 支持自定义各层剪裁率：意为是否支持手动指定各个卷积层的剪裁率。


除了以上内置策略，PaddleSlim还支持用户自定义卷积通道剪裁策略，请参考：[自定义卷积通道剪裁教程](https://paddleslim.readthedocs.io/zh_CN/latest/tutorials/pruning/dygraph/self_defined_filter_pruning.html)

## 各类方法效果对比

| 模型 | 压缩方法 | 精度(Top-1/Top-5) | 模型体积（MB） | GFLOPs |PaddleLite推理耗时|
|:--:|:---:|:--:|:--:|:--:|:--:|
| MobileNetV1 |Baseline                      |70.99%/89.68%                 |17|1.11      |66.052\35.8014\19.5762|
| MobileNetV1 |uniform + L1NormFilterPruner  |69.40%/88.66% (-1.59%/-1.02%) |9 |0.56(-50%)|33.5636\18.6834\10.5076|
| MobileNetV1 |sensitive + L1NormFilterPruner|70.4%/89.3% (-0.59%/-0.38%)   |12|0.74(-30%)| 46.5958\25.3098\13.6982|
| MobileNetV1 |sensitive + L1NormFilterPruner|69.8%/88.9% (-1.19%/-0.78%)   |9 |0.56(50%) |37.9892\20.7882\11.3144|
| MobileNetV1 |uniform + FPGMFilterPruner    |69.56%/89.14% (-1.43%/-0.53%) |9 |0.56(-50%)|33.5636\18.6834\10.5076|

注：
- uniform： 各层剪裁率保持一样。
- sensitive： 根据各层敏感度确定每层的剪裁率。


## 策略介绍

### L1NormFilterPruner

paper: https://arxiv.org/abs/1608.08710

该策略使用`l1-norm`统计量来表示一个卷积层内各个`Filters`的重要性，`l1-norm`越大的`Filter`越重要。

使用方法如下：

#### 动态图

```
pruner =paddleslim. L1NormFilterPruner(net, [1, 3, 128, 128])
pruner.prune_vars({"conv2d_0.w_0": 0.3})
```

[API文档](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/dygraph/pruners/l1norm_filter_pruner.html) | [完整示例](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/dygraph/dygraph_pruning_tutorial.html)

#### 静态图

```
pruner = paddleslim.prune.Pruner(criterion='l1_norm')
pruned_program, _, _ = pruner.prune(
        train_program,
        fluid.global_scope(),
        params=["conv2d_0.w_0"],
        ratios=[0.3],
        place=fluid.CPUPlace())
```

[API文档](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/prune/prune_api.html) | [完整示例](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/static/pruning_tutorial.html)


### FPGMFilterPruner

论文: https://arxiv.org/abs/1811.00250

该策略通过统计`Filters`两两之间的几何距离来评估单个卷积内的`Filters`的重要性。直觉上理解，离其它`Filters`平均距离越远的`Filter`越重要。

使用方法如下：

#### 动态图

```
pruner =paddleslim.FPGMFilterPruner(net, [1, 3, 128, 128])
pruner.prune_vars({"conv2d_0.w_0": 0.3})
```

[API文档](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/dygraph/pruners/fpgm_filter_pruner.html) | [完整示例](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/dygraph/dygraph_pruning_tutorial.html)

#### 静态图

```
pruner = paddleslim.prune.Pruner(criterion='geometry_median')
pruned_program, _, _ = pruner.prune(
        train_program,
        fluid.global_scope(),
        params=["conv2d_0.w_0"],
        ratios=[0.3],
        place=fluid.CPUPlace())
```

[API文档](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/prune/prune_api.html) | [完整示例](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/static/pruning_tutorial.html)


### SlimFilterPruner

论文: https://arxiv.org/pdf/1708.06519.pdf

该策略根据卷积之后的`batch_norm`的`scales`来评估当前卷积内各个`Filters`的重要性。`scale`越大，对应的`Filter`越重要。

使用方法如下：

#### 静态图

```
pruner = paddleslim.prune.Pruner(criterion='bn_scale')
pruned_program, _, _ = pruner.prune(
        train_program,
        fluid.global_scope(),
        params=["conv2d_0.w_0"],
        ratios=[0.3],
        place=fluid.CPUPlace())
```

[API文档](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/prune/prune_api.html) | [完整示例](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/static/pruning_tutorial.html)


### OptSlimFilterPruner

论文: https://arxiv.org/pdf/1708.06519.pdf

使用方法如下：

#### 静态图

```
pruner = paddleslim.prune.Pruner(criterion='bn_scale', idx_selector="optimal_threshold")
pruned_program, _, _ = pruner.prune(
        train_program,
        fluid.global_scope(),
        params=["conv2d_0.w_0"],
        ratios=[0.3],
        place=fluid.CPUPlace())
```

[API文档](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/prune/prune_api.html) | [完整示例](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/static/pruning_tutorial.html)
