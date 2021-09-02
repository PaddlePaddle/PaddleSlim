# 非结构化稀疏 -- GMP训练方式介绍与示例

## 简介

承接一步到位的稀疏化训练方式（即根据预训练模型只做一次剪裁，再去`finetune`），我们为`PaddleSlim`引入了一种逐步增加稀疏度的训练方式：`GMP`（`Gradual Magnitude Pruning`）。详细介绍可以参考[博客](https://neuralmagic.com/blog/pruning-gmp/)。这个训练策略的引入是为了解决大稀疏度训练时，精度恢复困难的问题。最终实验证明，`MobileNetV1-ImageNet`任务，在75%稀疏化conv1x1的实验中，使用`GMP`算法会使模型精度提升1%以上。

## 具体介绍

概念上讲，由于网络的参数的重要性与绝对值大小不是严格的对应关系，且随着数值增大，该对应关系会越薄弱，所以一步剪裁会对某些绝对值大的权重不友好，进而影响稀疏化模型的精度。而`GMP`算法采用逐步增加稀疏度的训练方式，增加了权重空间的灵活性，可以使得权重在稀疏化训练中去做适应性优化，从而在一定程度上保证精度。`GMP`将稀疏化训练过程分为三个阶段：稳定训练（`stable phase`）、剪裁训练（`pruning phase`）和调优训练（`finetuning phase`）。三个阶段中逐步增加稀疏度，且有各自的学习率变化。

- 稳定阶段：该阶段epoch较小，用于正式剪裁前的稳定模型。我们测试来看，对于模型精度的影响较小。这是因为我们调用了`pretrained_model`。开发者可以根据需求自行调整，一般取0（有预训练模型）或者2~5（无预训练模型）即可。稀疏度保持为0、学习率保持初始数值。

- 剪裁阶段：该阶段的epoch/iteration数目等于全量训练的一半，且在该过程中，稀疏度从某一个初始值（`inital ratio`）增加到最终值（`target ratio`），且增加的幅度逐渐减小。数学上，`ratio`变化为：

  $ratio = ((i / pruning_steps) - 1.0) ^ 3 + 1$

  $ratio_scaled = initial_ratio + (target_ratio - initial_ratio) * ratio$

上述$ratio$为一个三次函数所得，在$i == pruning_steps$时，$ratio=1$且梯度为0，保证了稀疏度随训练增加且增加速度减小。其中，$pruning_steps$代表$ratio$增加的次数，一般每一个epoch增加2次即可，我们实验发现，当剪裁次数过多时，也会不利于精度恢复。$ratio_scaled$则是根据输入的初始稀疏度和目标稀疏度，对$ratio$进行缩放和平移所得。

此外，学习率在该过程中保持为初始数值，不衰减。

- 调优阶段：该阶段的epoch/iteration数目等于全量训练的一半，且在该过程中，稀疏度保持为最终值（`target ratio`）。学习率衰减。`piecewise_decay`方式时，将调优阶段等分，设置为衰减边界即可。

## 参数介绍

根据上一节的具体介绍，我们归纳参数及其设置规则如下：
- stable_epochs: 0 (pretrained_model) / 2-5 (from-scratch model)
- pruning_epochs: total_epochs / 2
- tunning_epochs: total_epochs / 2
- pruning_steps: pruning_epochs * 2
- initial_ratio: 0.15
- lr: 预训练时的一个中间lr即可。例如，`MobileNetV1-ImageNet`预训练时，学习率由0.1降低为0.0001，我们在稀疏化训练时就采用了0.005。
- learning_rate_strategy: 目前仅支持piecewise_decay。cosine_decay的方式正在开发中。
- piecewise_decay_bound: $stable_epochs+pruning_epochs+tunning_epochs/3$, $stable_epochs+pruning_epochs+2*tunning_epochs/3$

## 代码调用
本节介绍如何在静态图和动态图中方便的调用`GMP`训练策略，以达到保证精度的目标。

```python
# 将上述参数定义为配置字典
configs = {
    'stable_iterations': args.stable_epochs * step_per_epoch,
    'pruning_iterations': args.pruning_epochs * step_per_epoch,
    'tunning_iterations': args.tunning_epochs * step_per_epoch,
    'resume_iteration': (args.last_epoch + 1) * step_per_epoch,
    'pruning_steps': args.pruning_steps,
    'initial_ratio': args.initial_ratio,
}

# 将configs作为参数初始化GMPUnstructuredPruner即可。
# 静态图
pruner = GMPUnstructuredPruner(
    train_program,
    mode='ratio', # 模式必须为'ratio'，'threshold'模式与GMP不兼容。
    ratio=args.ratio,
    place=place,
    configs=configs)

# 动态图
pruner = GMPUnstructuredPruner(
    model,
    mode='ratio', # 模式必须为'ratio'，'threshold'模式与GMP不兼容。
    ratio=args.ratio,
    configs=configs)
```

后续调用与正常训练无异，请参照[静态图](./README.md)和[动态图](../dygraph/unstructured_pruning/README.md)


## 实验结果
| 模型 | 数据集 | 压缩方法 | 压缩率| Top-1/Top-5 Acc | lr | threshold | epoch |
|:--:|:---:|:--:|:--:|:--:|:--:|:--:|:--:|
| MobileNetV1 | ImageNet | Baseline | - | 70.99%/89.68% | - | - | - |
| MobileNetV1 | Imagenet | ratio, 1x1conv      | 75% | 68.76%/88.91% (-2.23%/-0.77%) | 0.005 | - | 108 |
| MobileNetV1 | Imagenet | ratio, 1x1conv, GMP | 75% | 70.49%/89.48% (-0.50%/-0.20%) | 0.005 | - | 108 |
| MobileNetV1 | Imagenet | ratio, 1x1conv, GMP | 80% | 70.02%/89.26% (-0.97%/-0.42%) | 0.005 | - | 108 |
