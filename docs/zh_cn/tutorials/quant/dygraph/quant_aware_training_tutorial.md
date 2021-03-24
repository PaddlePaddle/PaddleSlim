# 量化训练详细教程

在线量化是在模型训练的过程中建模定点量化对模型的影响，通过在模型计算图中插入量化节点，在训练建模量化对模型精度的影响降低量化损失。

> 注意：目前动态图量化训练还不支持有控制流逻辑的模型，如果量化训练中出现Warning，推荐使用静态图量化训练功能。

PaddleSlim包含`QAT量化训练`和`PACT改进的量化训练`两种量化方法

- QAT
- PACT

## 使用方法

在线量化的基本流程可以分为以下四步：

1. 选择量化配置
2. 转换量化模型
3. 启动量化训练
4. 保存量化模型

下面分别介绍以下几点：

### 1. 选择量化配置

首先我们需要对本次量化的一些基本量化配置做一些选择，例如weight量化类型，activation量化类型等。如果没有特殊需求，可以直接拷贝我们默认的量化配置。全部可选的配置可以参考PaddleSlim量化文档，例如我们用的量化配置如下：

```python
quant_config = {
    'weight_preprocess_type': None,
    'activation_preprocess_type': None,
    'weight_quantize_type': 'channel_wise_abs_max',
    'activation_quantize_type': 'moving_average_abs_max',
    'weight_bits': 8,
    'activation_bits': 8,
    'dtype': 'int8',
    'window_size': 10000,
    'moving_rate': 0.9,
    'quantizable_layer_type': ['Conv2D', 'Linear'],
}
```

### 2. 转换量化模型

在确认好我们的量化配置以后，我们可以根据这个配置把我们定义好的一个普通模型转换为一个模拟量化模型。我们根据量化原理中介绍的PACT方法，定义好PACT函数pact和其对应的优化器pact_opt。在这之后就可以进行转换，转换的方式也很简单:

```python
import paddleslim
quanter = paddleslim.QAT(config=quant_config)
quanter.quantize(net)
```

### 3. 启动量化训练

得到了量化模型后就可以启动量化训练了，量化训练与普通的浮点数模型训练并无区别，无需增加新的代码或逻辑，直接按照浮点数模型训练的流程进行即可。

### 4. 保存量化模型

量化训练结束后，我们需要对量化模型做一个转化。PaddleSlim会对底层的一些量化OP顺序做调整，以便预测使用。转换及保存的基本流程如下所示:

```python
import paddleslim
quanter.save_quantized_model(
  model,
  path,
  input_spec=[paddle.static.InputSpec()])
```

量化预测模型可以使用`netron`软件打开，进行可视化查看。该量化预测模型和普通FP32预测模型一样，可以使用PaddleLite和PaddleInference加载预测，具体请参考`推理部署`章节。

## PACT在线量化

PACT方法是对普通在线量化方法的改进，对于一些量化敏感的模型，例如MobileNetV3，PACT方法一般都能降低量化模型的精度损失。

使用方法上与普通在线量化方法相近：

```python
# 在quant_config中额外指定'weight_preprocess_type'为'PACT'
    quant_config = {
        'weight_preprocess_type': 'PACT',
        'weight_quantize_type': 'channel_wise_abs_max',
        'activation_quantize_type': 'moving_average_abs_max',
        'weight_bits': 8,
        'activation_bits': 8,
        'dtype': 'int8',
        'window_size': 10000,
        'moving_rate': 0.9,
        'quantizable_layer_type': ['Conv2D', 'Linear'],
    }

```

详细代码与例程请参考：

- [动态图量化训练](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/dygraph/quant)

## 实验结果

|       模型        |     压缩方法     | 原模型Top-1/Top-5 Acc | 量化模型Top-1/Top-5 Acc |
| :---------------: | :--------------: | :-------------------: | :---------------------: |
|    MobileNetV1    |   quant_aware    |     70.99%/89.65%     |      70.63%/89.65%      |
|    MobileNetV2    |   quant_aware    |     72.15%/90.65%     |      72.05%/90.63%      |
|     ResNet50      |   quant_aware    |     76.50%/93.00%     |      76.48%/93.11%      |
| MobileNetV3_large | pact_quant_aware |     78.96%/94.48%     |      77.52%/93.77%      |

## 参考文献

1. [Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/pdf/1806.08342.pdf)

2. [PACT: Parameterized Clipping Activation for Quantized Neural Networks](https://arxiv.org/abs/1805.06085)
