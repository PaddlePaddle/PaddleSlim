# 在线量化

在线量化是在模型训练的过程中建模定点量化对模型的影响，通过在模型计算图中插入量化节点实现，详细算法原理请参考：[量化训练]([https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/algo/algo.md#122-%E9%87%8F%E5%8C%96%E8%AE%AD%E7%BB%83](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/algo/algo.md#122-量化训练))

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
# 静态图
quant_config = {
    'weight_quantize_type': 'abs_max',
    'activation_quantize_type': 'moving_average_abs_max',
    'weight_bits': 8,
    'activation_bits': 8,
    'not_quant_pattern': ['skip_quant'],
    'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul'],
    'dtype': 'int8',
    'window_size': 10000,
    'moving_rate': 0.9
}
# 动态图
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
# 静态图
compiled_train_prog = paddleslim.quant.quant_aware(train_prog, place, quant_config, scope=None, for_test=False)
# 动态图
quanter = paddleslim.dygraph.quant.QAT(config=quant_config)
quanter.quantize(net)
```

### 3. 启动量化训练

得到了量化模型后就可以启动量化训练了，量化训练与普通的浮点数模型训练并无区别，无需增加新的代码或逻辑，直接按照浮点数模型训练的流程进行即可。

### 4. 保存量化模型

量化训练结束后，我们需要对量化模型做一个转化。PaddleSlim会对底层的一些量化OP顺序做调整，以便预测使用。转换及保存的基本流程如下所示:

```python
import paddleslim
# 静态图
qat_program = paddleslim.quant.convert(val_program, place, quant_config, scope=None)
paddle.static.save_inference_model(
  dirname=save_path,
  feeded_var_names=feed_var_names,
  target_vars=target_vars,
  executor=exe,
  main_program=qat_program)
# 动态图
quanter.save_quantized_model(
  model,
  path,
  input_spec=[paddle.static.InputSpec()])
```

## PACT在线量化

PACT方法是对普通在线量化方法的改进，对于一些量化敏感的模型，例如MobileNetV3，PACT方法一般都能降低量化模型的精度损失

使用方法上与普通在线量化方法相近：

```python
# 静态图
# 提前定义好PACT函数
def pact(x):
  helper = LayerHelper("pact", **locals())
  dtype = 'float32'
  init_thres = values[x.name.split('_tmp_input')[0]]
  u_param_attr = paddle.ParamAttr(
    name=x.name + '_pact',
    initializer=paddle.nn.initializer.Constant(value=init_thres),
    regularizer=paddle.regularizer.L2Decay(0.0001),
    learning_rate=1)
  u_param = helper.create_parameter(
    attr=u_param_attr, shape=[1], dtype=dtype)

  gamma = paddle.nn.functional.relu(x - u_param)
  beta = paddle.nn.functional.relu(-u_param - x)
  x = x - gamma + beta
  return x
def get_optimizer():
  return paddle.optimizer.Momentum(args.lr, 0.9)
# 额外传入act_preprocess_func和optimizer_func
compiled_train_prog = quant_aware(
  train_prog,
  place,
  quant_config,
  act_preprocess_func=pact,
  optimizer_func=get_optimizer,
  executor=executor,
  for_test=False)
# 动态图
# 只需在quant_config中额外指定'weight_preprocess_type'为'PACT'
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

- [静态图普通在线量化](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/quant/quant_aware)
- [静态图PACT在线量化](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/quant/pact_quant_aware)
- [动态图普通、PACT在线量化](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/dygraph/quant)

## 实验结果

|       模型        |     压缩方法     | 原模型Top-1/Top-5 Acc | 量化模型Top-1/Top-5 Acc |
| :---------------: | :--------------: | :-------------------: | :---------------------: |
|    MobileNetV1    |   quant_aware    |     70.99%/89.65%     |      70.63%/89.65%      |
|    MobileNetV2    |   quant_aware    |     72.15%/90.65%     |      72.05%/90.63%      |
|     ResNet50      |   quant_aware    |     76.50%/93.00%     |      76.48%/93.11%      |
| MobileNetV3_large | pact_quant_aware |     78.96%/94.48%     |      77.52%/93.77%      |
