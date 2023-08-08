#  离线量化

离线量化又称为训练后量化，仅需要使用少量校准数据，确定最佳的量化参数降低量化误差。这种方法需要的数据量较少，但量化模型精度相比在线量化稍逊。


## 使用方法

离线量化的基本流程可以分为以下三步：

1. 选择量化配置
2. 采样收集量化信息
3. 转换量化模型

## 接口介绍

### 1. 量化配置相关概念以及接口：

`Observer`：用于统计OP输入或输出，并计算出量化相关的统计量，比如scale、zero_point等。每个离线量化算法对应一个Observer，现已有的Observer包含：
- `AVGObserver`：收集目标Tensor的平均值作为量化scale
- `MSEObserver`：收集最大绝对值并通过最小化MSE误差，收集量化scale
- `EMDObserver`：收集最大绝对值并通过最小化EMD误差，收集量化scale
- `HistObserver`：将张量值收集到直方图中，并根据百分比计算量化scale
- `KLObserver`：以最小化浮点值分布与量化浮点值分布之间的 Kullback-Leibler散度计算量化scale
- `AbsMaxChannelWiseWeightObserver`：根据目标权重的通道维度，收集最大绝对值作为量化scale
- `MSEChannelWiseWeightObserver`：根据目标权重的通道维度，收集最大绝对值并通过最小化MSE误差，收集量化scale

`Quanter`：对OP的输入或输出执行量化或模拟操作操作，同时还可以对输入Tensor的数值进行统计分析。每个量化训练算法对应一个Quanter，现已有的Quanter包含：
- `PACTQuanter`
- `WeightLSQplusQuanter`：
- `ActLSQplusQuanter`

`QuantConfig`：在执行量化操作之前，首先要配置量化相关的信息，主要是指定每层的各个输入使用什么Observer或Quanter。可通过以下调用方法，根据需求添加每层的量化配置信息：
| **QuantConfig接口**  | **传入参数及其含义**                              | **注意事项**                              |
|-----------------------------|-----------------------------------------|-----------------------------------------|
| add_layer_config | `layer`: 指定模型的某一层或某些层的list<br><br>`activation`: 用于量化激活以上指定layer的`Observer`或`Quanter` <br><br> `weight`: 用于量化权重以上指定layer的`Observer`或`Quanter` | 此方法是最高优的要求，这些层的量化方式将按照这里的要求，而不是按照其他配置进行量化
| add_name_config | `layer_name`: 指定模型的某一层的名字或某些层的名字的list <br><br> `activation`: 用于量化激活以上指定layer的`Observer`或`Quanter` <br><br> `weight`: 用于量化权重以上指定layer的`Observer`或`Quanter` | 此方法的优先级仅此于add_layer_config
| add_type_config | `layer_type`：指定需要量化的layer类型，可以为单个layer类型，或一个layer类型的list，layer类型必须为paddle.nn.Layer的子类 <br><br> `activation`: 用于量化激活以上指定layer的`Observer`或`Quanter` <br><br> `weight`: 用于量化权重以上指定layer的`Observer`或`Quanter` | 此方法的优先级此于add_name_config，指定需要量化的layer类型，如nn.Linear, 量化时将对所有nn.Linear进行量化，并指定weight和activation的quanter类型
| add_qat_layer_mapping | `source`：被量化的layer <br><br> `target`：量化的layer | source和target必须为paddle.nn.Layer的子类；当指定需要量化的layer类型，如果在框架中没有实现该层量化时，需要指定该layer的量化层，比如ColumnParallelLinear对应PaddleSlim中实现的QuantizedColumnParallelLinear

### 2. PTQ接口介绍：
| **PTQ接口**  | **传入参数及其含义**                              | **介绍**                              |
|-----------------------------|-----------------------------------------|-----------------------------------------|
| quantize | `model`：需要被量化的模型 <br> `inplace`：inplace=True时，该模型会被inplace的量化；inplace=False时，不改变原模型，并且会return一个量化的模型 | 对模型需要量化的层插入observers以采样到需要的量化信息
| convert | `model`：需要被转化的量化模型 <br> `inplace`：inplace=True时，该模型会被inplace的量化；inplace=False时，不改变原模型，并且会return一个量化的模型 | 将模型转化成onnx形式，进行此步骤之后才能对量化模型进行验证、导出成静态图等


## 使用示例
```python
import paddle
import paddleslim
from paddle.vision.models import mobilenet_v1
from paddle.quantization import QuantConfig
from paddle.quantization import PTQ
from paddleslim.quant.observers import HistObserver, KLObserver, EMDObserver, MSEObserver, AVGObserver, MSEChannelWiseWeightObserver, AbsMaxChannelWiseWeightObserver

# create the model
model = mobilenet_v1()

# define QuantConfig
q_config = QuantConfig(activation=None, weight=None)

# define act_quanter and weight_quanter
act_quanter = MSEObserver()
weight_quanter = MSEObserver()

# map ColumnParallelLinear to QuantizedColumnParallelLinear
q_config.add_qat_layer_mapping(ColumnParallelLinear,
                                QuantizedColumnParallelLinear)
# map RowParallelLinear to QuantizedRowParallelLinear
q_config.add_qat_layer_mapping(RowParallelLinear,
                                QuantizedRowParallelLinear)
# for each layer if type in [paddle.nn.Linear, ColumnParallelLinear, RowParallelLinear]
# make them quantizable
q_config.add_type_config(
        [paddle.nn.Linear, ColumnParallelLinear, RowParallelLinear],
        activation=activation,
        weight=weight,
    )


ptq = PTQ(q_config)
model = ptq.quantize(model, inplace=True)

# ptq sample
ptq_step = 100
for step, data in enumerate(dataloader):
    pred = model(data)
    if step == ptq_step:
        break

# convert to quant model that can evaluate and export
model = ptq.convert(model, inplace=True)
```
