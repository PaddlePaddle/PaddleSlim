# 可微分架构搜索DARTS(Differentiable Architecture Search)方法使用示例

本示例介绍如何使用PaddlePaddle进行可微分架构搜索，可以直接使用[DARTS](https://arxiv.org/abs/1806.09055)和[PC-DARTS](https://arxiv.org/abs/1907.05737)两种方法，也支持自定义修改后使用其他可微分架构搜索算法。

## 依赖项

> PaddlePaddle >= 1.7.0

## 数据集

本示例使用`CIFAR10`数据集进行架构搜索，可选择在`CIFAR10`或`ImageNet`数据集上做架构评估。
`CIFAR10`数据集可以在进行架构搜索或评估的过程中自动下载, `ImageNet`数据集需要自行下载，可参照此[教程](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification#%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87)


## 网络结构搜索

搜索方法支持DARTS的一阶、二阶近似搜索方法和PC-DARTS的搜索方法:
``` bash
python search.py                       # DARTS一阶近似搜索方法
python search.py --unrolled=True       # DARTS的二阶近似搜索方法
python search.py --method='PC-DARTS'   # PC-DARTS搜索方法
```


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
