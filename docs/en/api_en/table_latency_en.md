# Table about hardware lantency

The table about hardware latency is used to evaluate the inference time in special environment and inference engine. The following text used to introduce the format that PaddleSlim support.

## Introduce

The table about hardware latency saved all possible operations, one operation in the table including type and parameters, such as: type can be `conv2d`, and corresponding parameters can be the size of feature map, number of kernel, and the size of kernel.
The latency of every operation depends on hardware and inference engine.

## Overview format
The table about hardware latency saved in the way of file or multi-line string.
The first line of the table about hardware latency saved the information about version, every line in the following represents a operation and its latency.

## Version

The information about version split by comma in the english format, and the detail is hardware, inference engine and timestamp.

- ** hardware: ** Used to mark the environment of hardware, including type of architecture, version and so on.

- ** inference engine: ** Used to mark inference engine, including the name of inference engine, version, optimize options and so on.

- ** timestamp: ** Used to mark the time of this table created.

## Operation

The information about operation split by comma in the english format, the information about operation and latency split by tabs.

### conv2d

**format**

```text
op_type,flag_bias,flag_relu,n_in,c_in,h_in,w_in,c_out,groups,kernel,padding,stride,dilation\tlatency
```

**introduce**

- **op_type(str)** - The type of this op.
- **flag_bias (int)** - Whether has bias or not(0: donot has bias, 1: has bias).
- **flag_relu (int)** - Whether has relu or not(0: donot has relu, 1: has relu).
- **n_in (int)** - The batch size of input.
- **c_in (int)** - The number of channel about input.
- **h_in (int)** - The height of input feature map.
- **w_in (int)** - The width of input feature map.
- **c_out (int)** - The number of channel about output.
- **groups (int)** - The group of conv2d.
- **kernel (int)** - The size of kernel.
- **padding (int)** - The size of padding.
- **stride (int)** - The size of stride.
- **dilation (int)** - The size of dilation.
- **latency (float)** - The latency of this op.

### activaiton

**format**

```text
op_type,n_in,c_in,h_in,w_in\tlatency
```

**introduce**

- **op_type(str)** - The type of this op.
- **n_in (int)** - The batch size of input.
- **c_in (int)** - The number of channel about input.
- **h_in (int)** - The height of input feature map.
- **w_in (int)** - The width of input feature map.
- **latency (float)** - The latency of this op.

### batch_norm

**format**

```text
op_type,active_type,n_in,c_in,h_in,w_in\tlatency
```

**introduce**

- **op_type(str)** - The type of this op.
- **active_type (string|None)** - The type of activation function, including relu, prelu, sigmoid, relu6, tanh.
- **n_in (int)** - The batch size of input.
- **c_in (int)** - The number of channel about input.
- **h_in (int)** - The height of input feature map.
- **w_in (int)** - The width of input feature map.
- **latency (float)** - The latency of this op.

### eltwise

**format**

```text
op_type,n_in,c_in,h_in,w_in\tlatency
```

**introduce**

- **op_type(str)** - The type of this op.
- **n_in (int)** - The batch size of input.
- **c_in (int)** - The number of channel about input.
- **h_in (int)** - The height of input feature map.
- **w_in (int)** - The width of input feature map.
- **latency (float)** - The latency of this op.

### pooling

**format**

```text
op_type,flag_global_pooling,n_in,c_in,h_in,w_in,kernel,padding,stride,ceil_mode,pool_type\tlatency
```

**introduce**

- **op_type(str)** - The type of this op.
- **flag_global_pooling (int)** - Whether is global pooling or not(0: is not global, 1: is global pooling).
- **n_in (int)** - The batch size of input.
- **c_in (int)** - The number of channel about input.
- **h_in (int)** - The height of input feature map.
- **w_in (int)** - The width of input feature map.
- **kernel (int)** - The size of kernel.
- **padding (int)** - The size of padding.
- **stride (int)** - The size of stride.
- **ceil_mode (int)** - Whether to compute height and width by using ceil function(0: use floor function, 1: use ceil function).
- **pool_type (int)** - The type of pooling(1: max pooling 2: average pooling including padding 3: average pooling excluding padding).
- **latency (float)** - The latency of this op.

### softmax

**format**

```text
op_type,axis,n_in,c_in,h_in,w_in\tlatency
```

**introduce**

- **op_type(str)** - The type of this op.
- **axis (int)** - The index to compute softmax, index in the range of [-1, rank-1], `rank` is the rank of input.
- **n_in (int)** - The batch size of input.
- **c_in (int)** - The number of channel about input.
- **h_in (int)** - The height of input feature map.
- **w_in (int)** - The width of input feature map.
- **latency (float)** - The latency of this op.
