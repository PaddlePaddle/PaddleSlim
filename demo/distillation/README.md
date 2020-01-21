# 知识蒸馏示例

本示例将介绍如何使用知识蒸馏接口训练模型，蒸馏训练得到的模型相比不使用蒸馏策略的基线模型在精度上会有一定的提升。

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

### 2. 启动训练

在配置好ImageNet数据集后，用以下命令启动训练即可:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python distill.py
```

### 3. 训练结果

对比不使用蒸馏策略的基线模型(Top-1/Top-5: 70.99%/89.68%)，

经过120轮的蒸馏训练，MobileNet模型的Top-1/Top-5准确率达到72.77%/90.68%, Top-1/Top-5性能提升+1.78%/+1.00%

详细实验数据请参见[PaddleSlim模型库蒸馏部分](https://paddlepaddle.github.io/PaddleSlim/model_zoo/#13)
