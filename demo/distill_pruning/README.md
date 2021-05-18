# 知识蒸馏+非结构稀疏化示例

本示例将介绍如何使用知识蒸馏接口训练模型，并在此基础上调用稀疏化接口，达到同事稀疏化和蒸馏的效果。

## 接口介绍

请参考 [知识蒸馏API文档](https://paddlepaddle.github.io/PaddleSlim/api/single_distiller_api/)。

### 1. 蒸馏训练配置

示例使用ResNet50_vd作为teacher模型，对MobileNet结构的student网络进行蒸馏训练。

默认配置:

```yaml
batch_size: 256
init_lr: 0.1
lr_strategy: piecewise_decay
l2_decay: 3e-5
momentum_rate: 0.9
num_epochs: 120
data: imagenet
```
训练使用默认配置启动即可

### 2. 非结构化稀疏配置

示例采用迭代的非结构化稀疏训练方式：前若干个（warm_epochs）epochs不加稀疏化，此后每隔（pruning_period）个epochs，增加一定量的稀疏度，直到最大稀疏度（ratio）。其中训练过程中，蒸馏loss始终存在。

默认配置：

```yaml
pruning_mode: ratio
ratio: 0.85
warm_epochs: 20
pruning_period: 20
```

### 2. 启动训练

在配置好ImageNet数据集后，用以下命令启动训练即可:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python distill.py
```

### 3. 训练结果

持续训练中。
