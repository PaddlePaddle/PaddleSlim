# 硬件延时评估表

硬件延时评估表用于快速评估一个模型在特定硬件环境和推理引擎上的推理速度。
该文档主要用于定义PaddleSlim支持的硬件延时评估表的格式。

## 概述

硬件延时评估表中存放着所有可能的操作对应的延时信息，该表中的一个操作包括操作类型和操作参数，比如：操作类型可以是`conv2d`，对应的操作参数有输入特征图的大小、卷积核个数、卷积核大小等。
给定操作的延时依赖于硬件环境和推理引擎。

## 整体格式

硬件延时评估表以文件或多行字符串的形式保存。

硬件延时评估表第一行保存版本信息，后续每行为一个操作和对应的延时信息。

## 版本信息

版本信息以英文字符逗号分割，内容依次为硬件环境名称、推理引擎名称和时间戳。

- **硬件环境名称：** 用于标识硬件环境，可以包含计算架构类型、版本号等信息。

- **推理引擎名称：** 用于标识推理引擎，可以包含推理引擎名称、版本号、优化选项等信息。

- **时间戳：** 该评估表的创建时间。

## 操作信息

操作信息字段之间以逗号分割。操作信息与延迟信息之间以制表符分割。

### conv2d

**格式**

```text
op_type,flag_bias,flag_relu,n_in,c_in,h_in,w_in,c_out,groups,kernel,padding,stride,dilation\tlatency
```

**字段解释**

- **op_type(str)** - 当前op类型。
- **flag_bias (int)** - 是否有 bias（0：无，1：有）。
- **flag_relu (int)** - 是否有 relu（0：无，1：有）。
- **n_in (int)** - 输入 Tensor 的批尺寸 (batch size)。
- **c_in (int)** - 输入 Tensor 的通道 (channel) 数。
- **h_in (int)** - 输入 Tensor 的特征高度。
- **w_in (int)** - 输入 Tensor 的特征宽度。
- **c_out (int)** - 输出 Tensor 的通道 (channel) 数。
- **groups (int)** - 卷积二维层（Conv2D Layer）的组数。
- **kernel (int)** - 卷积核大小。
- **padding (int)** - 填充 (padding) 大小。
- **stride (int)** - 步长 (stride) 大小。
- **dilation (int)** - 膨胀 (dilation) 大小。
- **latency (float)** - 当前op的延时时间

### activation

**格式**

```text
op_type,n_in,c_in,h_in,w_in\tlatency
```

**字段解释**

- **op_type(str)** - 当前op类型。
- **n_in (int)** - 输入 Tensor 的批尺寸 (batch size)。
- **c_in (int)** - 输入 Tensor 的通道 (channel) 数。
- **h_in (int)** - 输入 Tensor 的特征高度。
- **w_in (int)** - 输入 Tensor 的特征宽度。
- **latency (float)** - 当前op的延时时间

### batch_norm

**格式**

```text
op_type,active_type,n_in,c_in,h_in,w_in\tlatency
```

**字段解释**

- **op_type(str)** - 当前op类型。
- **active_type (string|None)** - 激活函数类型，包含：relu, prelu, sigmoid, relu6, tanh。
- **n_in (int)** - 输入 Tensor 的批尺寸 (batch size)。
- **c_in (int)** - 输入 Tensor 的通道 (channel) 数。
- **h_in (int)** - 输入 Tensor 的特征高度。
- **w_in (int)** - 输入 Tensor 的特征宽度。
- **latency (float)** - 当前op的延时时间

### eltwise

**格式**

```text
op_type,n_in,c_in,h_in,w_in\tlatency
```

**字段解释**

- **op_type(str)** - 当前op类型。
- **n_in (int)** - 输入 Tensor 的批尺寸 (batch size)。
- **c_in (int)** - 输入 Tensor 的通道 (channel) 数。
- **h_in (int)** - 输入 Tensor 的特征高度。
- **w_in (int)** - 输入 Tensor 的特征宽度。
- **latency (float)** - 当前op的延时时间

### pooling

**格式**

```text
op_type,flag_global_pooling,n_in,c_in,h_in,w_in,kernel,padding,stride,ceil_mode,pool_type\tlatency
```

**字段解释**

- **op_type(str)** - 当前op类型。
- **flag_global_pooling (int)** - 是否为全局池化（0：不是，1：是）。
- **n_in (int)** - 输入 Tensor 的批尺寸 (batch size)。
- **c_in (int)** - 输入 Tensor 的通道 (channel) 数。
- **h_in (int)** - 输入 Tensor 的特征高度。
- **w_in (int)** - 输入 Tensor 的特征宽度。
- **kernel (int)** - 卷积核大小。
- **padding (int)** - 填充 (padding) 大小。
- **stride (int)** - 步长 (stride) 大小。
- **ceil_mode (int)** - 是否用 ceil 函数计算输出高度和宽度。0 表示使用 floor 函数，1 表示使用 ceil 函数。
- **pool_type (int)** - 池化类型，其中 1 表示 pooling_max，2 表示 pooling_average_include_padding，3 表示 pooling_average_exclude_padding。
- **latency (float)** - 当前op的延时时间

### softmax

**格式**

```text
op_type,axis,n_in,c_in,h_in,w_in\tlatency
```

**字段解释**

- **op_type(str)** - 当前op类型。
- **axis (int)** - 执行 softmax 计算的维度索引，应该在 [−1，rank − 1] 范围内，其中 rank 是输入变量的秩。
- **n_in (int)** - 输入 Tensor 的批尺寸 (batch size)。
- **c_in (int)** - 输入 Tensor 的通道 (channel) 数。
- **h_in (int)** - 输入 Tensor 的特征高度。
- **w_in (int)** - 输入 Tensor 的特征宽度。
- **latency (float)** - 当前op的延时时间
